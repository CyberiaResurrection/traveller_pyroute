"""
Created on Jun 3, 2014

@author: tjoneslo
"""
import argparse
import codecs
import os
import time
import urllib.error
import urllib.parse
import urllib.request


def get_url(url, sector, suffix, output_dir) -> bool:
    try:
        f = urllib.request.urlopen(url)
    except urllib.error.HTTPError as ex:
        print("get URL failed: {} -> {}".format(url, ex))
        return False
    except urllib.error.URLError as ex:
        print("get URL failed: {} -> {}".format(url, ex))
        return False

    encoding = f.headers['content-type'].split('charset=')[-1]
    content = f.read()
    if encoding == 'text/xml' or encoding == 'text/html':
        ucontent = str(content, 'utf-8').replace('\r\n', '\n')
    else:
        ucontent = str(content, encoding).replace('\r\n', '\n')

    path = os.path.join(output_dir, '%s.%s' % (sector, suffix))

    with codecs.open(path, 'wb', 'utf-8') as out:
        out.write(ucontent)
    f.close()
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download sector/metadata from TravellerMap')
    parser.add_argument('--routes', dest='routes', default=False, action='store_true',
                        help='Include route information in the sector downloads')
    parser.add_argument('sector_list', help='List of sectors to download')
    parser.add_argument('output_dir', help='output directory for sector data and xml metadata')
    parser.add_argument('--milieu', default="M1105", help="Milieu of data to download")
    args = parser.parse_args()

    with codecs.open(args.sector_list, 'r', encoding="utf-8") as f:
        sectorsList = [line for line in f]

    for raw_sector in sectorsList:
        sector = raw_sector.rstrip()
        print('Downloading %s' % sector)
        params = {'sector': sector, 'type': 'SecondSurvey', "milieu": args.milieu}
        if args.routes:
            params['routes'] = '1'
        url_params = urllib.parse.urlencode(params)
        url = 'http://www.travellermap.com/api/sec?%s' % url_params

        success = get_url(url, sector, 'sec', args.output_dir)
        if not success:
            print("Retrying " + sector)
            get_url(url, sector, 'sec', args.output_dir)

        url_params = urllib.parse.urlencode({'sector': sector, 'accept': 'text/xml'})
        url = 'http://travellermap.com/api/metadata?%s' % url_params
        success = get_url(url, sector, 'xml', args.output_dir)
        if not success:
            print("Retrying XML for " + sector)
            get_url(url, sector, 'sec', args.output_dir)

        time.sleep(5)
