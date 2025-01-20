"""
Created on Jan 19, 2025

@author: CyberiaResurrection
"""
import os

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.colors import toColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from PyRoute.Outputs.Colour import Colour
from PyRoute.Outputs.Cursor import Cursor
from PyRoute.Outputs.FontLayer import FontLayer
from PyRoute.Outputs.Map import MapOutput, Scheme


class PDFMap(MapOutput):

    subsector_grid_width = 592
    subsector_width = 144
    subsector_grid_height = 780
    subsector_height = 180

    def __init__(self, galaxy, routes: str, output_path: str, writer: str):
        super(PDFMap, self).__init__(galaxy, routes, output_path, writer)
        self.doc: canvas.Canvas
        self.font_layer = FontLayer()

        zapf_path = self.font_layer.getpath('ZapfDingbats-Regular.ttf')
        pdfmetrics.registerFont(TTFont('Zapf', zapf_path))

        self.fonts: dict[str, [str, float]] = {
            'title': ('Times-Bold', 25),
            'info': ('Times-Roman', 7),
            'sector': ('Times-Roman', 10),
            'system_port': ('Helvetica-Bold', 5),
            'system_uwp': ('Helvetica-Bold', 3.5),
            'system_name': ('Times-Bold', 5.5),
            'base code': ('Zapf', 5)
        }

        self.colours: dict[str, Colour] = {
            'background': 'white',
            'title': 'black',
            'info': 'black',
            'sector': 'black',
            'system_port': 'black',
            'system_uwp': 'black',
            'system_name': 'black',
            'base code': 'black',

            'grid': 'lightgrey',
            'hexes': 'lightgrey',
            'red zone': 'crimson',
            'amber zone': 'goldenrod',
            'gg refuel': 'goldenrod',
            'wild refuel': 'lightblue',
            'comm': toColor('rgb(83, 204, 106)'),
            'trade': 'red'
        }
        self.logger.debug("Completed PDFMap init")

    def document(self, area_name: str, is_live=True):
        path = os.path.join(self.output_path, f"{area_name}.pdf")
        self.logger.info(f"writing PDF to {path}")

        self.doc = canvas.Canvas(filename=path, pagesize=LETTER, bottomup=False)
        self.doc.setCreator("PyRoute using ReportLab")
        self.doc.setSubject("Trade route map generated by PyRoute for Traveller")
        self.doc.setTitle(area_name)
        self.doc.setPageCompression(is_live)
        self.image_size = Cursor(LETTER[0], LETTER[1])
        # self.doc.set_margins(4)
        return self.doc

    def close(self):
        self.doc.showPage()
        self.doc.save()

    def add_line(self, start: Cursor, end: Cursor, colour: Colour, stroke: str = 'solid', width: float = 1) -> None:
        self.doc.setStrokeColor(colour)
        self.doc.setLineWidth(width)
        if stroke != 'solid':
            self.doc.setDash()
        self.doc.line(start.x, start.y, end.x, end.y)

    def add_rectangle(self, start: Cursor, end: Cursor, border_colour: Colour, fill_colour: Colour, width: int) -> None:
        if border_colour is None:
            border_colour = 'white'
        self.doc.setLineWidth(width)
        self.doc.setStrokeColor(border_colour)
        fill = False
        if fill_colour is not None:
            self.doc.setFillColor(fill_colour)
            fill = True

        self.doc.rect(start.x, start.y, end.x - start.x, end.y - start.y, width, fill)

    def add_circle(self, center: Cursor, radius: int, line_width: int, fill: bool, scheme: Scheme) -> None:
        if self.colours[scheme] is None:
            return
        self.doc.setStrokeColor(self.colours[scheme])
        self.doc.setLineWidth(line_width)

        if fill:
            self.doc.setFillColor(self.colours[scheme])
        self.doc.circle(center.x, center.y, radius, 1, 1 if fill else 0)

    def add_text(self, text: str, start: Cursor, scheme: Scheme):
        font_info = self.get_font(scheme)
        self.doc.setFont(font_info[0], size=font_info[1])
        self.doc.setFillColor(self.colours[scheme])
        self.doc.drawString(start.x, start.y, text)

    def add_text_centred(self, text: str, start: Cursor, scheme: Scheme, max_width: int = -1, offset: bool = False):
        font_info = self.get_font(scheme)
        self.doc.setFont(font_info[0], size=font_info[1])
        self.doc.setFillColor(self.colours[scheme])
        out_text = text
        offset_x = 0
        if max_width > 0 and len(text) > 0:
            for chars in range(len(text), 0, -1):
                width = self.doc.stringWidth(text[:chars], font_info[0], font_info[1])
                if width <= max_width:
                    out_text = text[:chars]
                    if offset and width // 2 != int(width) / 2:
                        offset_x = 1.0
                    break
        elif len(text) > 0:
            width = self.doc.stringWidth(text, font_info[0], font_info[1])
            if offset and width // 2 != int(width) / 2:
                offset_x = 1.0

        self.doc.drawCentredString(start.x + offset_x, start.y + font_info[1], out_text)

    def add_text_rotated(self, text: str, start: Cursor, scheme: Scheme, rotation: int) -> None:
        self.doc.saveState()
        font_info = self.get_font(scheme)
        self.doc.setFont(font_info[0], size=font_info[1])
        self.doc.setFillColor(self.colours[scheme])
        self.doc.rotate(-rotation)

        if rotation > 0:
            # Spinward Side
            self.doc.drawCentredString(-start.y, start.x, text)
        else:
            # Trailing side
            self.doc.drawCentredString(start.y, -start.x, text)

        self.doc.restoreState()

    def add_text_right_aligned(self, text: str, start: Cursor, scheme: Scheme) -> None:
        font_info = self.get_font(scheme)
        self.doc.setFont(font_info[0], size=font_info[1])
        self.doc.setFillColor(self.colours[scheme])
        width = self.doc.stringWidth(text, font_info[0], font_info[1])
        self.doc.drawString(start.x - width, start.y, text)

    @staticmethod
    def _get_colour(colour: Colour) -> Colour:
        return colour
