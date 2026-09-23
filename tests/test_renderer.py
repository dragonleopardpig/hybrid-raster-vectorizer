import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from hybrid_vectorizer.renderer import load_spec, render_file, render_svg


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "examples" / "interference" / "spec.json"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


class RendererTest(unittest.TestCase):
    def test_example_contains_semantic_vector_geometry(self) -> None:
        content = render_svg(load_spec(SPEC_PATH))
        root = ET.fromstring(content)
        paths = root.findall(f".//{SVG_NAMESPACE}path")
        lines = root.findall(f".//{SVG_NAMESPACE}line")
        images = root.findall(f".//{SVG_NAMESPACE}image")
        texts = root.findall(f".//{SVG_NAMESPACE}text")

        curve = next(path for path in paths if path.attrib.get("class") == "curve")
        self.assertEqual(curve.attrib["d"].count("C"), 14)
        self.assertEqual(len(lines), 12)
        self.assertEqual(images, [])
        self.assertEqual(len(texts), 23)

    def test_render_file_writes_valid_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.svg"
            render_file(SPEC_PATH, output_path)
            parsed = ET.parse(output_path)
            self.assertEqual(parsed.getroot().tag, f"{SVG_NAMESPACE}svg")


if __name__ == "__main__":
    unittest.main()
