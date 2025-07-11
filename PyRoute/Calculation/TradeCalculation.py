"""
Created on Mar 15, 2014

@author: tjoneslo
"""
import functools
import itertools
import math
from typing import Optional

import numpy as np
import networkx as nx

from PyRoute.Pathfinding.DistanceGraph import DistanceGraph
from PyRoute.Allies.AllyGen import AllyGen
from PyRoute.Calculation.RouteCalculation import RouteCalculation
try:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnified import ApproximateShortestPathForestUnified
except ModuleNotFoundError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified  # type: ignore
except ImportError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified  # type: ignore
from PyRoute.TradeBalance import TradeBalance
try:
    from PyRoute.Pathfinding.astar_numpy import astar_path_numpy
except ModuleNotFoundError:
    from PyRoute.Pathfinding.astar_numpy_fallback import astar_path_numpy
except ImportError:
    from PyRoute.Pathfinding.astar_numpy_fallback import astar_path_numpy
from PyRoute.Star import Star


class TradeCalculation(RouteCalculation):
    """
    Perform the trade calculations by generating the routes
    between all the trade pairs
    """
    # Weight for route over a distance. The relative cost for
    # moving freight between two worlds a given distance apart
    # in a single jump.
    # These are made up from whole cloth.
    # distance_weight = [0, 30, 50, 70, 90, 120, 140 ]

    # GT Weights based upon one pass estimate
    # distance_weight = [0, 30, 50, 70, 110, 170, 300]

    # Pure HG weights
    # distance_weight = [0, 30, 50, 75, 130, 230, 490]

    # MGT weights
    # distance_weight = [0, 30, 60, 105, 190, 410, 2470]

    # T5 Weights, now with Hop Drive
    distance_weight = [0, 30, 50, 75, 130, 230, 490, 9999, 9999, 9999, 300]

    # max_connections = [6, 18, 36, 60, 90, 126, 168, 216, 270, 330]
    max_connections = [6, 12, 18, 30, 45, 63, 84, 108, 135, 165]

    # Set an initial range for influence for worlds based upon their
    # wtn. For a given world look up the range given by (wtn-8) (min 0),
    # and the system checks every other world in that range for trade
    # opportunity. See the btn_jump_mod and min btn to see how
    # worlds are excluded from this list.
    btn_range = [2, 9, 29, 59, 99, 299, 599]

    # Maximum WTN to process routes for
    max_wtn = 15

    def __init__(self, galaxy, min_btn=13, route_btn=8, route_reuse=10, debug_flag=False):
        super(TradeCalculation, self).__init__(galaxy)

        # Minimum BTN to calculate routes for. BTN between two worlds less than
        # this value are ignored. Set lower to have more routes calculated, but
        # may not have have an impact on the overall trade flows.
        self.min_btn = min_btn
        self.min_route_wtn = (min_btn - 1) // 2  # In light of minimum btn, what is smallest WTN that _can_ meet it?

        # Minimum WTN to process routes for
        self.min_wtn = route_btn

        # Override the default setting for route-reuse from the base class
        # based upon program arguments.
        self.route_reuse = route_reuse
        # Testing indicated that allowing a little more than 1 edge hit before tripping an update seemed to
        # strike the best space/time tradeoff, so default epsilon to sqrt(10/route_reuse).  The 0.1 cap is to speed up
        # the default case.
        self.epsilon = 0.1 * min(1, (10 / route_reuse) ** 0.5)

        # Are debugging gubbins turned on?
        self.debug_flag = debug_flag
        self.pathfinding_data = None

        # Count routes that get trimmed by as-found route length
        self.penumbra_routes = 0

        self.shortest_path_tree = None
        # Track inter-sector passenger imbalances
        self.sector_passenger_balance = TradeBalance(stat_field="passengers", region=galaxy)
        # Track inter-sector trade imbalances
        self.sector_trade_balance = TradeBalance(stat_field="tradeExt", region=galaxy, target="trade")
        # Track inter-sector trade volume imbalances
        self.sector_trade_volume_balance = TradeBalance(stat_field="tradeDtonExt", region=galaxy, target="trade")
        # Track inter-allegiance passenger imbalances
        self.allegiance_passenger_balance = TradeBalance(stat_field="passengers", region=galaxy, field="alg",
                                                         star_field="allegiance_base", target_property="code")
        # Track inter-allegiance trade imbalances
        self.allegiance_trade_balance = TradeBalance(stat_field="trade", region=galaxy, field="alg",
                                                     star_field="allegiance_base", target_property="code")
        # Track inter-allegiance trade imbalances
        self.allegiance_trade_volume_balance = TradeBalance(stat_field="tradeDtonExt", region=galaxy, field="alg",
                                                     star_field="allegiance_base", target_property="code")

    def base_route_filter(self, star, neighbor) -> bool:
        # by the time we've _reached_ here, we're assuming generate_base_routes() has handled the unilateral filtering
        # - in this case, red/forbidden zones and barren systems - so only bilateral filtering remains.
        return False

    def base_range_routes(self, star, neighbor) -> Optional[int]:
        dist = star.distance(neighbor)
        # add all the stars in the BTN range, but skip this pair
        # if there there isn't enough trade to warrant a trade check
        btn = self.get_btn(star, neighbor, dist)
        if btn >= self.min_btn:
            passBTN = self.get_passenger_btn(btn, star, neighbor)
            self.galaxy.ranges.add_edge(star, neighbor, distance=dist, btn=btn, passenger_btn=passBTN)

        return None if dist > self.galaxy.max_jump_range else dist

    @functools.cache
    def _max_dist(self, star_wtn, neighbour_wtn, maxjump=False):
        if neighbour_wtn < star_wtn:
            return self._max_dist(neighbour_wtn, star_wtn, maxjump)
        max_dist = self.btn_range[min(max(0, neighbour_wtn - self.min_wtn), 6)]
        if maxjump:
            return max(max_dist, self.galaxy.max_jump_range)
        return max_dist

    def _raw_ranges(self):
        max_route_dist = max(self.btn_range)
        max_range = self.galaxy.max_jump_range
        min_btn = self.min_btn
        min_wtn = self.min_route_wtn

        hiball = [item for item in self.galaxy.ranges if item.wtn >= min_wtn and not item.is_redzone]
        loball = [item for item in self.galaxy.ranges if item.wtn < min_wtn and not item.is_redzone]

        ranges = [(star, neighbour) for (star, neighbour) in itertools.combinations(hiball, 2)
                  if (dist := star.distance(neighbour)) <= self._max_dist(star.wtn, neighbour.wtn, True)
                  and self._get_btn_upper_bound(star, neighbour, max_range, min_btn, distance=dist) >= min_btn
                  ]
        lo_ranges = [(star, neighbour) for (star, neighbour) in itertools.combinations(loball, 2)
                     if (star.distance(neighbour)) <= max_range
                     ]
        mid_ranges = [(star, neighbour) for (star, neighbour) in itertools.product(hiball, loball)
                      if (star.distance(neighbour)) <= max_range
                      ]
        ranges.extend(lo_ranges)
        ranges.extend(mid_ranges)
        self.logger.info("Routes with endpoints more than " + str(max_route_dist) + " pc apart, trimmed")

        return ranges

    def generate_routes(self) -> None:
        """
        Generate the basic routes between all the stars. This creates two sets
        of routes.
        - Stars: The basic J4 (max-jump) routes for all pairs of stars.
        - Ranges: The set of trade routes needing to be calculated.
        """
        self.generate_base_routes()

        self.logger.info('calculating routes...')
        self.galaxy.is_well_formed()
        for star in self.galaxy.stars:
            if len(self.galaxy.stars[star]) < 11:
                continue
            neighbor_routes = [(s, n, d) for (s, n, d) in self.galaxy.stars.edges([star], True)]
            # Need to do two sorts here:
            # BTN low to high to find them first
            # Range high to low to find them first
            neighbor_routes.sort(key=lambda tn: tn[2]['btn'])
            neighbor_routes.sort(key=lambda tn: tn[2]['distance'], reverse=True)

            length = len(neighbor_routes)

            # remove edges from the list which are
            # A) The most distant first
            # B) The lowest BTN for equal distant routes
            # If the neighbor has only a few (<15) connections don't remove that one
            # until there are 20 connections left.
            # This may be reduced by other stars deciding you are too far away.
            for (s, n, d) in neighbor_routes:
                if len(self.galaxy.stars[n]) < 15:
                    continue
                if length <= self.max_connections[self.galaxy.max_jump_range - 1]:
                    break
                if d.get('xboat', False) or d.get('comm', False):
                    continue
                self.galaxy.stars.remove_edge(s, n)
                length -= 1

        self.logger.info('Final route count {}'.format(self.galaxy.stars.number_of_edges()))

    def calculate_routes(self) -> None:
        """
        The base calculate routes. Read through all the stars in WTN order.
        Do this order to allow the higher trade routes establish the basic routes
        for the lower routes to follow.
        """
        self.logger.info('sorting routes...')
        # Filter out pathfinding attempts that can never return a route, as they're between two different
        # connected components in the underlying galaxy.stars graph - such pathfinding attempts are doomed
        # to failure.
        self.calculate_components()

        btn_skipped = [(s, n) for (s, n) in self.galaxy.ranges.edges() if s.component != n.component]
        self.logger.info(f"Found {len(btn_skipped)} non-component routes, removing from ranges graph")
        for s, n in btn_skipped:
            self.galaxy.ranges.remove_edge(s, n)
        self.logger.info(f"Removed {len(btn_skipped)} non-component routes from ranges graph")

        btn = [(s, n, d) for (s, n, d) in self.galaxy.ranges.edges(data=True)]
        btn.sort(key=lambda tn: tn[2]['btn'], reverse=True)
        if self.debug_flag:
            self.pathfinding_data = {'nodes_expanded': np.ones(len(btn), dtype=float) * -1,
                                     'nodes_queued': np.ones(len(btn), dtype=float) * -1,
                                     'branch_factor': np.ones(len(btn), dtype=float) * -1,
                                     'nodes_revisited': np.ones(len(btn), dtype=float) * -1,
                                     'neighbour_bound': np.ones(len(btn), dtype=float) * -1,
                                     'new_upbounds': np.ones(len(btn), dtype=float) * -1,
                                     'g_exhausted': np.ones(len(btn), dtype=float) * -1,
                                     'f_exhausted': np.ones(len(btn), dtype=float) * -1,
                                     'targ_exhausted': np.ones(len(btn), dtype=float) * -1,
                                     'un_exhausted': np.ones(len(btn), dtype=float) * -1,
                                     'neighbourhood_size': np.ones(len(btn), dtype=float) * -1}

        # Pick landmarks - biggest WTN system in each graph component.  It worked out simpler to do this for _all_
        # components, even those with only one star.
        self.logger.info("Finding pathfinding landmarks")
        self.star_graph = DistanceGraph(self.galaxy.stars)
        self.logger.info("Generating pathfinding landmarks")
        landmarks, self.component_landmarks = self.get_landmarks(index=True, btn=btn)
        self.logger.info("Pathfinding landmarks found")

        source = max(self.galaxy.star_mapping.values(), key=lambda item: item.wtn)
        source.is_landmark = True
        # Feed the landmarks in as roots of their respective shortest-path trees.
        # This sets up the approximate-shortest-path bounds to be during the first pathfinding call.
        self.shortest_path_tree = ApproximateShortestPathForestUnified(source.index, self.galaxy.stars,
                                                                             self.epsilon, sources=landmarks)
        self.star_len_root = max(1, math.floor(math.sqrt(len(self.star_graph))) // 2)

        base_btn = 0
        counter = 0
        processed = 0
        total = len(btn)
        for (star, neighbor, data) in btn:
            if base_btn != data['btn']:
                if counter > 0:
                    self.logger.info('processed {} routes at BTN {}'.format(counter, base_btn))
                base_btn = data['btn']
                counter = 0
            if total > 100 and processed % (total // 20) == 0:
                self.logger.info('processed {} routes, at {}%'.format(processed, processed // (total // 100)))
            self.get_trade_between(star, neighbor)
            counter += 1
            processed += 1
        self.multilateral_balance_trade()
        self.multilateral_balance_pass()
        self.logger.info('processed {} routes at BTN {}'.format(counter, base_btn))
        self.logger.info('{} penumbra routes included out of {}'.format(self.penumbra_routes, processed))
        if self.debug_flag:
            num_stars = len(self.galaxy.stars)
            self.logger.info('Pathfinding diagnostic data for route reuse {}, {} stars, {} routes'.
                             format(self.route_reuse, num_stars, processed))
            keep = self.pathfinding_data['nodes_expanded'] != -1
            branchdata = self.pathfinding_data['branch_factor']
            branchdata = branchdata[1 <= branchdata]
            branchdata = branchdata[branchdata < float('+inf')]
            branch = np.percentile(branchdata, [50, 80, 98])
            branch_geomean = round(10 ** np.mean(np.log10(branchdata)), 3)
            neighbourhood_size = np.round(np.percentile(self.pathfinding_data['neighbourhood_size'], [50, 80, 98]), 3)
            total_expanded = int(np.sum(self.pathfinding_data['nodes_expanded'][keep]))
            total_queued = int(np.sum(self.pathfinding_data['nodes_queued'][keep]))
            total_revisited = int(np.sum(self.pathfinding_data['nodes_revisited'][keep]))
            total_neighbour_bound = int(np.sum(self.pathfinding_data['neighbour_bound'][keep]))
            total_upbounds = int(np.sum(self.pathfinding_data['new_upbounds'][keep]))
            total_g_exhausted = int(np.sum(self.pathfinding_data['g_exhausted'][keep]))
            total_f_exhausted = int(np.sum(self.pathfinding_data['f_exhausted'][keep]))
            total_targ_exhausted = int(np.sum(self.pathfinding_data['targ_exhausted'][keep]))
            total_un_exhausted = int(np.sum(self.pathfinding_data['un_exhausted'][keep]))
            self.logger.info('50th/80th/98th percentile effective branch factor {}/{}/{}'.format(branch[0], branch[1], branch[2]))
            self.logger.info('Geometric mean effective branch factor {}'.format(branch_geomean))
            self.logger.info('50th/80th/98th percentile neighbourhood size {}/{}/{}'.format(neighbourhood_size[0], neighbourhood_size[1], neighbourhood_size[2]))
            self.logger.info('Total nodes popped {}'.format(total_expanded))
            self.logger.info('Total nodes queued {}'.format(total_queued))
            self.logger.info('Total nodes revisited {}'.format(total_revisited))
            self.logger.info('Total neighbour bound checks {}'.format(total_neighbour_bound))
            self.logger.info('Total new upper bounds {}'.format(total_upbounds))
            self.logger.info('Total g-exhausted nodes {}'.format(total_g_exhausted))
            self.logger.info('Total f-exhausted nodes {}'.format(total_f_exhausted))
            self.logger.info('Total target-exhausted nodes {}'.format(total_targ_exhausted))
            self.logger.info('Total un-exhausted nodes {}'.format(total_un_exhausted))

    def get_trade_between(self, star, target) -> None:
        """
        Calculate the route between star and target
        If we can't find a route (no Jump 4 (or N) path), skip this pair
        otherwise update the trade information.
        """
        assert 'actual distance' not in self.galaxy.ranges._adj[target][star],\
            "This route from " + str(star) + " to " + str(target) + " has already been processed in reverse"

        try:
            # Get upper bound value, and increase by 0.5% to ensure it _is_ an upper bound
            upbound = self._preheat_upper_bound(star.index, target.index, allow_reheat=True) * 1.005

            comp_id = star.component
            if star.index in self.component_landmarks[comp_id] and \
                    target.index not in self.component_landmarks[comp_id]:
                target, star = star, target

            rawroute, diag = astar_path_numpy(self.star_graph, star.index, target.index,
                                              self.shortest_path_tree.lower_bound_bulk, upbound=upbound,
                                              diagnostics=self.debug_flag)

            if self.debug_flag:
                moshdex = np.where(self.pathfinding_data['branch_factor'] == -1.0)[0][0]
                # Now load up this route's summary data
                self.pathfinding_data['nodes_expanded'][moshdex] = diag['nodes_expanded']
                self.pathfinding_data['nodes_queued'][moshdex] = diag['nodes_queued']
                self.pathfinding_data['branch_factor'][moshdex] = diag['branch_factor']
                self.pathfinding_data['nodes_revisited'][moshdex] = diag['nodes_revisited']
                self.pathfinding_data['neighbour_bound'][moshdex] = diag['neighbour_bound']
                self.pathfinding_data['new_upbounds'][moshdex] = diag['new_upbounds']
                self.pathfinding_data['g_exhausted'][moshdex] = diag['g_exhausted']
                self.pathfinding_data['f_exhausted'][moshdex] = diag['f_exhausted']
                self.pathfinding_data['targ_exhausted'][moshdex] = diag['targ_exhausted']
                self.pathfinding_data['un_exhausted'][moshdex] = diag['un_exhausted']
                neighbourhood_size = 1 if diag['un_exhausted'] == 0 else diag['nodes_queued'] / diag['un_exhausted']
                self.pathfinding_data['neighbourhood_size'][moshdex] = neighbourhood_size

        except nx.NetworkXNoPath:
            return

        route = [self.galaxy.star_mapping[item] for item in rawroute]

        distance = self.route_distance(route)
        btn = self.get_btn(star, target, distance)

        if self.min_btn > btn:
            self.penumbra_routes += 1
            return

        if self.debug_flag:
            fwd_weight = self.route_cost(route)
            route.reverse()
            rev_weight = self.route_cost(route)
            route.reverse()
            delta = fwd_weight - rev_weight
            assert 1e-16 > delta * delta,\
                "Route weight between " + str(star) + " and " + str(target) + " should not be direction sensitive.  Forward weight " + str(fwd_weight) + ", rev weight " + str(rev_weight) + ", delta " + str(abs(delta))

        # Update the trade route (edges)
        tradeCr, tradePass, tradeDton = self.route_update_simple(route, True, distance=distance)
        self.update_statistics(star, target, tradeCr, tradePass, tradeDton)

    def _preheat_upper_bound(self, stardex, targdex, allow_reheat=True) -> float:
        # Don't reheat on _every_ route, but reheat frequently enough to keep historic costs sort-of firm.
        # Keeping this deterministic helps keep input reduction straight, as there's less state to track.
        reheat = allow_reheat and ((stardex + targdex) % (self.star_len_root) == 0)

        upbound = self.shortest_path_tree.triangle_upbound(stardex, targdex)

        # Case 0 - Source and target are directly connected
        keep = self.star_graph._arcs[stardex][0] == targdex
        if keep.any():
            flip = self.star_graph._arcs[stardex][1][keep]
            return min(upbound, flip[0])

        # Grab arrays to support Case 1
        hist_targ = self.galaxy.historic_costs._arcs[targdex]
        hist_src = self.galaxy.historic_costs._arcs[stardex]

        # Case 1 - Historic-route source neighbour to historic-route target neighbour
        if 0 < len(hist_src[0]) and 0 < len(hist_targ[0]):
            # Dig out the common neighbours, _and_ their indexes in the respective adjacency lists
            common, src, trg = np.intersect1d(hist_src[0], hist_targ[0], assume_unique=True, return_indices=True)
            common_len = len(common)
            if 0 < common_len:
                midbound = hist_src[1][src] + hist_targ[1][trg]
                mindex = np.argmin(midbound)
                upbound = min(upbound, midbound[mindex])
                if reheat:
                    adj = self.galaxy.stars._adj
                    reheat_list = set()
                    reheat_list.add((stardex, common[mindex]))
                    reheat_list.add((targdex, common[mindex]))
                    if 1 < common_len:
                        maxdex = np.argmax(midbound)
                        reheat_list.add((stardex, common[maxdex]))
                        reheat_list.add((targdex, common[maxdex]))

                    for pair in reheat_list:
                        edge = adj[pair[0]][pair[1]]

                        # The 0.5% bump is to _ensure_ the newcost remains an _upper_ bound
                        # on the historic-route cost
                        newcost = self.galaxy.route_cost(edge['route']) * 1.005
                        if edge['weight'] > newcost:
                            edge['weight'] = newcost
                            self.galaxy.historic_costs.lighten_edge(pair[0], pair[1], newcost)
                    reheated_upbound = self._preheat_upper_bound(stardex, targdex, allow_reheat=False)
                    upbound = min(reheated_upbound, upbound)

        return upbound

    def update_statistics(self, star, target, tradeCr, tradePass, tradeDton=0) -> None:
        if star.sector != target.sector:
            star.sector.stats.tradeExt += tradeCr // 2
            target.sector.stats.tradeExt += tradeCr // 2
            star.sector.subsectors[star.subsector()].stats.tradeExt += tradeCr // 2
            target.sector.subsectors[target.subsector()].stats.tradeExt += tradeCr // 2
            star.sector.stats.passengers += tradePass // 2
            target.sector.stats.passengers += tradePass // 2
            star.sector.stats.tradeDtonExt += tradeDton // 2
            target.sector.stats.tradeDtonExt += tradeDton // 2
            if 1 == (tradeCr & 1):
                self.sector_trade_balance.log_odd_unit(star, target)
            if 1 == (tradePass & 1):
                self.sector_passenger_balance.log_odd_unit(star, target)
            if 1 == (tradeDton & 1):
                self.sector_trade_volume_balance.log_odd_unit(star, target)
        else:
            star.sector.stats.trade += tradeCr
            star.sector.stats.passengers += tradePass
            star.sector.stats.tradeDton += tradeDton
            if star.subsector() == target.subsector():
                star.sector.subsectors[star.subsector()].stats.trade += tradeCr
                star.sector.subsectors[star.subsector()].stats.tradeDton += tradeDton
            else:
                star.sector.subsectors[star.subsector()].stats.tradeExt += tradeCr // 2
                target.sector.subsectors[target.subsector()].stats.tradeExt += tradeCr // 2
                star.sector.subsectors[star.subsector()].stats.tradeDtonExt += tradeDton // 2
                target.sector.subsectors[target.subsector()].stats.tradeDtonExt += tradeDton // 2
        starcode = AllyGen.same_align(star.alg_code)
        targcode = AllyGen.same_align(target.alg_code)
        # By definition, _any_ nonalighed system is _not_ allied to anything - even nonaligned systems of the same
        # allegiance code (eg NaVa, NaHu, etc).  As we're tracking allegiance-level imbalances, we can't just _ignore_
        # odd passengers/trade units.  The simplest way around this is to directly add the odd unit in to that
        # allegiance's tradeExt or passenger totals, as needed.
        double_up = AllyGen.is_nonaligned(starcode) and (starcode == targcode)

        if AllyGen.are_allies(star.alg_code, target.alg_code):
            self.galaxy.alg[starcode].stats.trade += tradeCr
            self.galaxy.alg[starcode].stats.passengers += tradePass
            self.galaxy.alg[starcode].stats.tradeDton += tradeDton
        else:
            self.galaxy.alg[starcode].stats.tradeExt += tradeCr // 2
            self.galaxy.alg[targcode].stats.tradeExt += tradeCr // 2
            self.galaxy.alg[starcode].stats.passengers += tradePass // 2
            self.galaxy.alg[targcode].stats.passengers += tradePass // 2
            self.galaxy.alg[starcode].stats.tradeDtonExt += tradeDton // 2
            self.galaxy.alg[targcode].stats.tradeDtonExt += tradeDton // 2
            if 1 == (tradeCr & 1):
                if double_up:
                    self.galaxy.alg[starcode].stats.tradeExt += 1
                else:
                    self.allegiance_trade_balance.log_odd_unit(star, target)
            if 1 == (tradePass & 1):
                if double_up:
                    self.galaxy.alg[starcode].stats.passengers += 1
                else:
                    self.allegiance_passenger_balance.log_odd_unit(star, target)
            if 1 == (tradeDton & 1):
                if double_up:
                    self.galaxy.alg[starcode].stats.tradeDtonExt += 1
                else:
                    self.allegiance_trade_volume_balance.log_odd_unit(star, target)

        self.galaxy.stats.trade += tradeCr
        self.galaxy.stats.passengers += tradePass
        self.galaxy.stats.tradeDton += tradeDton

        try:
            if 0 == (star.index + target.index) % (self.star_len_root):
                self.cross_check_totals()
        except AssertionError as e:
            msg = str(star.name) + "-" + str(target.name) + ": " + str(e)
            raise AssertionError(msg)

    @staticmethod
    @functools.cache
    def _balance_tuple(name_from, name_to):
        if name_from <= name_to:
            return (name_from, name_to)
        return (name_to, name_from)

    def route_update_simple(self, route, reweight=True, distance=None) -> tuple[int, int, int]:
        """
        Update the trade calculations based upon the route selected.
        - add the trade values for the worlds, and edges
        - add a count for the worlds and edges
        - reduce the weight of routes used to allow more trade to flow
        """
        distance = distance if isinstance(distance, int) else self.route_distance(route)

        source = route[0]
        target = route[-1]

        # Internal statistics
        rangedata = self.galaxy.ranges._adj[source][target]
        rangedata['actual distance'] = distance
        rangedata['jumps'] = len(route) - 1

        if 5 < len(route) and not (source.index in self.galaxy.stars and target.index in self.galaxy.stars[source.index]):
            cost = self.route_cost(route)
            self.galaxy.stars.add_edge(source.index, target.index, distance=distance, weight=cost, trade=0, btn=0,
                                       count=0, exhaust=0, route=route)
            self.galaxy.historic_costs.add_edge(source.index, target.index, cost)

        # Gather basic statistics.
        tradeBTN = self.get_btn(source, target, distance)
        tradeDton = self.calc_trade_tonnage(tradeBTN, distance)
        tradeCr = self.calc_trade(tradeBTN)
        source.tradeIn += tradeCr // 2
        target.tradeIn += tradeCr // 2
        tradePassBTN = self.get_passenger_btn(tradeBTN, source, target)
        tradePass = self.calc_passengers(tradePassBTN)

        source.passIn += tradePass
        target.passIn += tradePass

        edges = []
        start = source
        for end in route[1:-1]:
            end.tradeOver += tradeCr
            end.tradeCount += 1
            end.passOver += tradePass

        for end in route[1:]:
            data = self.galaxy.stars._adj[start.index][end.index]
            # exhausted = data['count'] >= data['exhaust']
            if reweight and (data['count'] < data['exhaust']):
                data['weight'] -= (data['weight'] - data['distance']) / self.route_reuse
                self.star_graph.lighten_edge(start.index, end.index, data['weight'])
                self.shortest_path_tree.lighten_edge(start.index, end.index, data['weight'])
                # Edge can only trip an update if it's not exhausted
                edges.append((start.index, end.index))
                data['count'] += 1
            data['trade'] += tradeCr
            start = end

        # Feed the list of touched edges into the approximate-shortest-path machinery, so it can update whatever
        # distance labels it needs to stay within its approximation bound.
        if reweight and 0 < len(edges):
            self.shortest_path_tree.update_edges(edges)

        return (tradeCr, tradePass, tradeDton)

    @staticmethod
    def route_distance(route) -> int:
        """
        Given a route, return its line length in parsec
        """
        distance = 0
        links = zip(route[0:-1], route[1:])
        for item in links:
            distance += item[0].distance(item[1])
        return distance

    def route_cost(self, route) -> float:
        """
        Given a route, return its total cost _at the moment_
        """
        return self.galaxy.route_cost(route)

    def route_weight(self, star, target) -> float:
        dist = star.distance(target)
        weight = self.distance_weight[dist]
        if target.alg_code != star.alg_code:
            weight += 25
        if star.port in 'CDEX?' or target.port in 'CDEX?':
            weight += 25
        if star.port in 'DEX?' or target.port in 'DEX?':
            weight += 25
        if star.deep_space_station or target.deep_space_station:
            weight += 100
        weight -= star.importance + target.importance
        # Per https://www.baeldung.com/cs/dijkstra-vs-a-pathfinding , to ensure termination in finite time:
        # "the edges have strictly positive costs"
        assert 0 < weight, "Weight of edge between " + str(star) + " and " + str(
            target) + " must be positive"
        return weight

    def unilateral_filter(self, star: Star) -> bool:
        if star.zone in ['R', 'F']:
            return True
        return star.tradeCode.barren

    def is_sector_trade_balanced(self) -> None:
        self.sector_trade_balance.is_balanced()

    def is_sector_pass_balanced(self) -> None:
        self.sector_passenger_balance.is_balanced()

    def is_sector_trade_volume_balanced(self) -> None:
        self.sector_trade_volume_balance.is_balanced()

    def is_allegiance_trade_balanced(self) -> None:
        self.allegiance_trade_balance.is_balanced()

    def is_allegiance_pass_balanced(self) -> None:
        self.allegiance_passenger_balance.is_balanced()

    def is_allegiance_trade_volume_balanced(self) -> None:
        self.allegiance_trade_volume_balance.is_balanced()

    def multilateral_balance_trade(self) -> None:
        self.sector_trade_balance.multilateral_balance()
        self.allegiance_trade_balance.multilateral_balance()

    def multilateral_balance_pass(self) -> None:
        self.sector_passenger_balance.multilateral_balance()
        self.allegiance_passenger_balance.multilateral_balance()

    def multilateral_balance_trade_volume(self) -> None:
        self.sector_trade_volume_balance.multilateral_balance()
        self.allegiance_trade_volume_balance.multilateral_balance()

    def cross_check_totals(self) -> None:
        grand_total_pax = self.galaxy.stats.passengers
        grand_total_trade = self.galaxy.stats.trade
        grand_total_volume = self.galaxy.stats.tradeDton

        total_sector_pax = sum([item.stats.passengers for item in self.galaxy.sectors.values()])
        total_sector_trade = sum([item.stats.trade + item.stats.tradeExt for item in self.galaxy.sectors.values()])
        total_sector_volume = sum([item.stats.tradeDton + item.stats.tradeDtonExt
                                   for item in self.galaxy.sectors.values()])

        base_allegiances = {AllyGen.same_align(item) for item in self.galaxy.alg}

        total_allegiance_pax = sum([self.galaxy.alg[item].stats.passengers for item in base_allegiances])
        total_allegiance_trade = sum([self.galaxy.alg[item].stats.trade + self.galaxy.alg[item].stats.tradeExt
                                      for item in base_allegiances])
        total_allegiance_volume = sum([self.galaxy.alg[item].stats.tradeDton + self.galaxy.alg[item].stats.tradeDtonExt
                                      for item in base_allegiances])

        assert grand_total_pax == total_sector_pax + self.sector_passenger_balance.sum, "Sector total pax not balanced with galaxy pax"
        assert grand_total_trade == total_sector_trade + self.sector_trade_balance.sum, "Sector total trade not balanced with galaxy trade"
        assert grand_total_volume == total_sector_volume + self.sector_trade_volume_balance.sum, "Sector total trade volume not balanced with galaxy trade volume"
        assert grand_total_pax == total_allegiance_pax + self.allegiance_passenger_balance.sum, "Allegiance total pax " + str(total_allegiance_pax + self.allegiance_passenger_balance.sum) + " not balanced with galaxy pax " + str(grand_total_pax)
        assert grand_total_trade == total_allegiance_trade + self.allegiance_trade_balance.sum, "Allegiance total trade not balanced with galaxy trade"
        assert grand_total_volume == total_allegiance_volume + self.allegiance_trade_volume_balance.sum, "Allegiance total trade volume not balanced with galaxy trade volume"
