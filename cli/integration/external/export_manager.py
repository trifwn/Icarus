"""Export manager for external tool format conversion."""

import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from .models import ConversionJob
from .models import ExportFormat
from .models import ExportFormatType
from .models import ExportResult


class ExportManager:
    """Manages export and format conversion for external tools."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._converters = {}
        self._active_jobs: Dict[str, ConversionJob] = {}
        self._setup_converters()

    def _setup_converters(self):
        """Set up format-specific converters."""
        self._converters = {
            ExportFormatType.JSON: self._export_json,
            ExportFormatType.CSV: self._export_csv,
            ExportFormatType.XML: self._export_xml,
            ExportFormatType.HDF5: self._export_hdf5,
            ExportFormatType.MATLAB: self._export_matlab,
            ExportFormatType.EXCEL: self._export_excel,
            ExportFormatType.PARAVIEW: self._export_paraview,
            ExportFormatType.TECPLOT: self._export_tecplot,
        }

    def get_supported_formats(self) -> List[ExportFormat]:
        """Get list of supported export formats."""
        formats = [
            ExportFormat(
                type=ExportFormatType.JSON,
                extension=".json",
                mime_type="application/json",
                supports_compression=True,
            ),
            ExportFormat(
                type=ExportFormatType.CSV,
                extension=".csv",
                mime_type="text/csv",
                supports_compression=True,
            ),
            ExportFormat(
                type=ExportFormatType.XML,
                extension=".xml",
                mime_type="application/xml",
                supports_compression=True,
            ),
            ExportFormat(
                type=ExportFormatType.HDF5,
                extension=".h5",
                mime_type="application/x-hdf5",
                supports_compression=True,
                max_file_size=10 * 1024 * 1024 * 1024,  # 10GB
            ),
            ExportFormat(
                type=ExportFormatType.MATLAB,
                extension=".mat",
                mime_type="application/x-matlab-data",
            ),
            ExportFormat(
                type=ExportFormatType.EXCEL,
                extension=".xlsx",
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            ExportFormat(
                type=ExportFormatType.PARAVIEW,
                extension=".vtu",
                mime_type="application/xml",
            ),
            ExportFormat(
                type=ExportFormatType.TECPLOT,
                extension=".plt",
                mime_type="application/octet-stream",
            ),
        ]
        return formats

    def get_format_by_type(
        self,
        format_type: ExportFormatType,
    ) -> Optional[ExportFormat]:
        """Get export format by type."""
        formats = self.get_supported_formats()
        for fmt in formats:
            if fmt.type == format_type:
                return fmt
        return None

    def get_format_by_extension(self, extension: str) -> Optional[ExportFormat]:
        """Get export format by file extension."""
        extension = extension.lower()
        if not extension.startswith("."):
            extension = "." + extension

        formats = self.get_supported_formats()
        for fmt in formats:
            if fmt.extension == extension:
                return fmt
        return None

    async def export_data(
        self,
        data: Any,
        output_path: Path,
        format_type: ExportFormatType,
        options: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """Export data to specified format."""
        options = options or {}

        try:
            # Get format info
            export_format = self.get_format_by_type(format_type)
            if not export_format:
                return ExportResult(
                    success=False,
                    errors=[f"Unsupported export format: {format_type}"],
                )

            # Get converter
            converter = self._converters.get(format_type)
            if not converter:
                return ExportResult(
                    success=False,
                    errors=[f"No converter available for {format_type}"],
                )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Perform conversion
            await converter(data, output_path, options)

            # Get file size
            file_size = output_path.stat().st_size if output_path.exists() else 0

            result = ExportResult(
                success=True,
                output_path=output_path,
                format=export_format,
                file_size=file_size,
            )

            self.logger.info(
                f"Successfully exported data to {output_path} ({format_type.value})",
            )
            return result

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return ExportResult(success=False, errors=[str(e)])

    async def _export_json(self, data: Any, output_path: Path, options: Dict[str, Any]):
        """Export data to JSON format."""
        # Convert data to JSON-serializable format
        json_data = self._prepare_json_data(data)

        indent = options.get("indent", 2)
        ensure_ascii = options.get("ensure_ascii", False)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_data,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                default=str,
            )

    async def _export_csv(self, data: Any, output_path: Path, options: Dict[str, Any]):
        """Export data to CSV format."""
        delimiter = options.get("delimiter", ",")
        include_header = options.get("include_header", True)

        if isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            # Convert list to DataFrame
            df = pd.DataFrame(data)
        elif hasattr(data, "to_dataframe"):
            # Custom object with DataFrame conversion
            df = data.to_dataframe()
        else:
            # Try to convert directly
            df = pd.DataFrame([data])

        df.to_csv(
            output_path,
            sep=delimiter,
            index=False,
            header=include_header,
            encoding="utf-8",
        )

    async def _export_xml(self, data: Any, output_path: Path, options: Dict[str, Any]):
        """Export data to XML format."""
        root_name = options.get("root_name", "data")
        item_name = options.get("item_name", "item")

        root = ET.Element(root_name)

        if isinstance(data, dict):
            self._dict_to_xml(data, root)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_elem = ET.SubElement(root, item_name)
                item_elem.set("index", str(i))
                if isinstance(item, dict):
                    self._dict_to_xml(item, item_elem)
                else:
                    item_elem.text = str(item)
        else:
            root.text = str(data)

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element):
        """Convert dictionary to XML elements."""
        for key, value in data.items():
            elem = ET.SubElement(parent, str(key))
            if isinstance(value, dict):
                self._dict_to_xml(value, elem)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    item_elem = ET.SubElement(elem, "item")
                    item_elem.set("index", str(i))
                    if isinstance(item, dict):
                        self._dict_to_xml(item, item_elem)
                    else:
                        item_elem.text = str(item)
            else:
                elem.text = str(value)

    async def _export_hdf5(self, data: Any, output_path: Path, options: Dict[str, Any]):
        """Export data to HDF5 format."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py package required for HDF5 export")

        compression = options.get("compression", "gzip")
        compression_opts = options.get("compression_opts", 6)

        with h5py.File(output_path, "w") as f:
            if isinstance(data, dict):
                self._dict_to_hdf5(data, f, compression, compression_opts)
            elif isinstance(data, (list, np.ndarray)):
                f.create_dataset(
                    "data",
                    data=np.array(data),
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                f.create_dataset("data", data=[data])

    def _dict_to_hdf5(
        self,
        data: Dict[str, Any],
        group,
        compression: str,
        compression_opts: int,
    ):
        """Convert dictionary to HDF5 groups and datasets."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._dict_to_hdf5(value, subgroup, compression, compression_opts)
            elif isinstance(value, (list, np.ndarray)):
                group.create_dataset(
                    key,
                    data=np.array(value),
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                group.create_dataset(key, data=[value])

    async def _export_matlab(
        self,
        data: Any,
        output_path: Path,
        options: Dict[str, Any],
    ):
        """Export data to MATLAB format."""
        try:
            from scipy.io import savemat
        except ImportError:
            raise ImportError("scipy package required for MATLAB export")

        # Convert data to MATLAB-compatible format
        matlab_data = self._prepare_matlab_data(data)

        savemat(str(output_path), matlab_data, do_compression=True)

    def _prepare_matlab_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for MATLAB export."""
        if isinstance(data, dict):
            return {
                k: np.array(v) if isinstance(v, list) else v for k, v in data.items()
            }
        elif isinstance(data, list):
            return {"data": np.array(data)}
        else:
            return {"data": data}

    async def _export_excel(
        self,
        data: Any,
        output_path: Path,
        options: Dict[str, Any],
    ):
        """Export data to Excel format."""
        sheet_name = options.get("sheet_name", "Sheet1")
        include_index = options.get("include_index", False)

        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif hasattr(data, "to_dataframe"):
            df = data.to_dataframe()
        else:
            df = pd.DataFrame([data])

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=include_index)

    async def _export_paraview(
        self,
        data: Any,
        output_path: Path,
        options: Dict[str, Any],
    ):
        """Export data to ParaView VTU format."""
        # This is a simplified implementation
        # In practice, you'd use libraries like meshio or vtk

        root = ET.Element("VTKFile")
        root.set("type", "UnstructuredGrid")
        root.set("version", "0.1")
        root.set("byte_order", "LittleEndian")

        unstructured_grid = ET.SubElement(root, "UnstructuredGrid")
        piece = ET.SubElement(unstructured_grid, "Piece")

        if isinstance(data, dict) and "points" in data:
            points_data = data["points"]
            piece.set("NumberOfPoints", str(len(points_data)))

            points = ET.SubElement(piece, "Points")
            data_array = ET.SubElement(points, "DataArray")
            data_array.set("type", "Float32")
            data_array.set("NumberOfComponents", "3")
            data_array.set("format", "ascii")

            # Convert points to string
            points_str = " ".join([f"{p[0]} {p[1]} {p[2]}" for p in points_data])
            data_array.text = points_str

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    async def _export_tecplot(
        self,
        data: Any,
        output_path: Path,
        options: Dict[str, Any],
    ):
        """Export data to Tecplot format."""
        title = options.get("title", "ICARUS Export")
        variables = options.get("variables", ["X", "Y", "Z"])

        with open(output_path, "w") as f:
            # Write header
            f.write(f'TITLE = "{title}"\n')
            f.write(f'VARIABLES = {", ".join([f'"{var}"' for var in variables])}\n')

            if isinstance(data, dict):
                # Assume data contains arrays for each variable
                num_points = len(next(iter(data.values())))
                f.write(f'ZONE T="Zone1", I={num_points}, F=POINT\n')

                for i in range(num_points):
                    values = [str(data[var][i]) for var in variables if var in data]
                    f.write(" ".join(values) + "\n")
            elif isinstance(data, list):
                f.write(f'ZONE T="Zone1", I={len(data)}, F=POINT\n')
                for item in data:
                    if isinstance(item, (list, tuple)):
                        f.write(" ".join(map(str, item)) + "\n")
                    else:
                        f.write(str(item) + "\n")

    def _prepare_json_data(self, data: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._prepare_json_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_json_data(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, "__dict__"):
            return self._prepare_json_data(data.__dict__)
        else:
            return data

    async def convert_format(
        self,
        input_path: Path,
        output_path: Path,
        target_format: ExportFormatType,
        options: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """Convert file from one format to another."""
        try:
            # Load data from input file
            data = await self._load_data(input_path)

            # Export to target format
            return await self.export_data(data, output_path, target_format, options)

        except Exception as e:
            self.logger.error(f"Format conversion failed: {e}")
            return ExportResult(success=False, errors=[str(e)])

    async def _load_data(self, input_path: Path) -> Any:
        """Load data from file based on extension."""
        extension = input_path.suffix.lower()

        if extension == ".json":
            with open(input_path, encoding="utf-8") as f:
                return json.load(f)
        elif extension == ".csv":
            return pd.read_csv(input_path).to_dict("records")
        elif extension == ".xlsx":
            return pd.read_excel(input_path).to_dict("records")
        elif extension == ".xml":
            tree = ET.parse(input_path)
            return self._xml_to_dict(tree.getroot())
        else:
            # Try to read as text
            with open(input_path, encoding="utf-8") as f:
                return f.read()

    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}

        # Add attributes
        if element.attrib:
            result["@attributes"] = element.attrib

        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:
                return element.text.strip()
            result["#text"] = element.text.strip()

        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result

    async def batch_export(
        self,
        data_items: List[Dict[str, Any]],
        output_dir: Path,
        format_type: ExportFormatType,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[ExportResult]:
        """Export multiple data items concurrently."""
        tasks = []

        for i, data in enumerate(data_items):
            filename = f"export_{i:04d}"
            export_format = self.get_format_by_type(format_type)
            if export_format:
                filename += export_format.extension

            output_path = output_dir / filename
            task = self.export_data(data, output_path, format_type, options)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        export_results = []
        for result in results:
            if isinstance(result, Exception):
                export_results.append(ExportResult(success=False, errors=[str(result)]))
            else:
                export_results.append(result)

        return export_results

    def get_conversion_job(self, job_id: str) -> Optional[ConversionJob]:
        """Get conversion job by ID."""
        return self._active_jobs.get(job_id)

    def list_active_jobs(self) -> List[ConversionJob]:
        """List all active conversion jobs."""
        return list(self._active_jobs.values())
