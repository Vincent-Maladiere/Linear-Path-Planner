from pykml import parser
import networkx as nx

from src.utils.logs import logger
from src.parsing.parcel import ParcelGroup, ParcelItem, Parcel


class GoogleEarthParserError(Exception):
    pass


class GoogleEarthParser:
    def __init__(self, workdir) -> None:

        self.workdir = workdir
        self.point_ref = None
        self.dict_parcel = dict()
        self.graph_parcel = nx.Graph()

    def get_parcel_group(self, filename_coords: str) -> ParcelGroup:
        """
        Create the parcel group object.

        1. Extract item from the parsed KML file, create parcel and add items to them.
        2. Create the ParcelGroup object, which represent all parcel data, and remove small obstacles.
        3. Save each parcels into the workdir.
        """

        logger.info(" # [GoogleEarthParser] get_parcels")

        name_doc, placemarks = fetch_file(filename_coords)
        for placemark in placemarks:
            if self.visible(placemark):
                parcel_name, item_name, item_type = self.extract_type(placemark)
                coords_global = self.extract_coords_global(placemark)

                if item_type == ParcelItem.gate:
                    self.add_gate_to_parcel(item_name, coords_global)
                else:
                    self.add_item_to_parcel(
                        parcel_name, item_name, item_type, coords_global
                    )

        parcel_group = ParcelGroup(
            self.workdir,
            name=str(name_doc),
            dict_parcel=self.dict_parcel,
            graph_parcel=self.graph_parcel,
        )

        return parcel_group

    def add_gate_to_parcel(self, item_name, coords_global):

        item_name_shorten = item_name.replace("gate ", "")
        parcel_names = [
            f"parcel {idx.strip()}" for idx in item_name_shorten.split("-")
        ]

        self.graph_parcel.add_edge(*parcel_names)

        for parcel_name in parcel_names:
            if parcel_name != "parcel start":
                self.add_item_to_parcel(
                    parcel_name, item_name, ParcelItem.gate, coords_global
                )

    def add_item_to_parcel(
        self, parcel_name, item_name, item_type, coords_global, verbose=False
    ):
        if verbose:
            logger.info(f" # [GoogleEarthParser] add {item_name} to {parcel_name}")

        if parcel_name not in self.dict_parcel:
            self.dict_parcel[parcel_name] = Parcel(parcel_name)

        self.dict_parcel[parcel_name].add_item(item_type, coords_global, self.point_ref)

    def visible(self, placemark):
        return getattr(placemark, "visibility", None) is None

    def extract_type(self, placemark):
        p_name = str(placemark.name).lower().replace("  ", " ")

        if not ("parcel" in p_name or ParcelItem.gate in p_name):
            raise GoogleEarthParserError(f"No 'parcel' or {ParcelItem.gate} in {p_name}")

        if "parcel" in p_name:
            parcel_name, item_name = p_name.split("-")
            parcel_name = parcel_name.strip()
            item_name = item_name.strip()

            for item in ParcelItem:
                if item in item_name:
                    return parcel_name, item_name, item
            else:
                raise GoogleEarthParserError(f"Unknown item type '{p_name}'")

        elif ParcelItem.gate in p_name:
            return None, p_name, ParcelItem.gate

        else:
            raise GoogleEarthParserError(
                f"Neither 'parcel' nor '{ParcelItem.gate}' in '{placemark.name}'"
            )

    def extract_coords_global(self, placemark):

        # TODO: handle Points

        if "Polygon" in dir(placemark):
            coordinates = str(placemark.Polygon.outerBoundaryIs.LinearRing.coordinates)
        else:
            coordinates = str(placemark.LineString.coordinates)

        coordinates = coordinates.strip().split()

        coords_global = []
        for coord in coordinates:
            lon, lat, _ = coord.split(",")
            coords_global.append((float(lon), float(lat)))

        if self.point_ref is None:
            self.point_ref = coords_global[0]  # lon, lat

        return coords_global


def fetch_file(filename_kml):

    with open(filename_kml, "rb") as f:
        file_kml = parser.fromstring(f.read())
    name_doc = file_kml.Document.name

    return name_doc, file_kml.Document.Placemark


def main():
    parser = GoogleEarthParser()
    parcel_group = parser.get_parcel_group("prod/MobileFence.kml")
    parcel_group.plot_parcels()
