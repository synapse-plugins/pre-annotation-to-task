import os
from urllib.parse import urlparse

import requests
from synapse_sdk.utils.converters.coco.to_dm import COCOToDMConverter
from synapse_sdk.utils.converters.dm.to_v1 import DMV2ToV1Converter
from synapse_sdk.utils.converters.pascal.to_dm import PascalToDMConverter
from synapse_sdk.utils.converters.yolo.to_dm import YOLOToDMConverter


class AnnotationToTask:
    def __init__(self, run, *args, **kwargs):
        """Initialize the plugin task pre annotation action class.

        Args:
            run: Plugin run object.
        """
        self.run = run

    def convert_data_from_file(
        self,
        primary_file_url: str,
        primary_file_original_name: str,
        data_file_url: str,
        data_file_original_name: str,
    ) -> dict:
        """Convert the data from a file to a task object.

        Args:
            primary_file_url (str): primary file url.
            primary_file_original_name (str): primary file original name.
            data_file_url (str): data file url.
            data_file_original_name (str): data file original name.

        Returns:
            dict: The converted data.
        """
        # Branch logics by Schema to convert (yolo, coco, pascal, dm_schema_ver_1, dm_schema_ver_2)
        schema_to_convert = self.run.context.get('params').get('pre_processor_params').get('schema_to_convert')

        project_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        temp_path = f'{project_base_path}/temp/'

        if schema_to_convert == 'yolo':
            yolo_class_names = self.run.context.get('params').get('pre_processor_params').get('yolo_class_names')
            yolo_class_names = yolo_class_names.split(',')
            converted_data = self._convert_yolo(primary_file_url, data_file_url, temp_path, yolo_class_names)
        elif schema_to_convert == 'coco':
            converted_data = self._convert_coco(
                primary_file_url, primary_file_original_name, data_file_url, data_file_original_name, temp_path
            )
        elif schema_to_convert == 'pascal':
            converted_data = self._convert_pascal(primary_file_url, data_file_url, temp_path)
        elif schema_to_convert == 'dm_schema_ver_1':
            converted_data = self._convert_dm_schema_v1(data_file_url)
        elif schema_to_convert == 'dm_schema_ver_2':
            converted_data = self._convert_dm_schema_v2(primary_file_url, data_file_url, temp_path)
        else:
            raise ValueError(f'Unsupported schema_to_convert: {schema_to_convert}')

        return converted_data

    def _convert_yolo(self, primary_file_url: str, data_file_url: str, temp_path: str, class_names: list) -> dict:
        """Convert YOLO format to DM format."""
        temp_primary_file_path = f'{temp_path}/images/'
        temp_data_file_path = f'{temp_path}/labels/'

        # Create temp_path directory if not exists
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(temp_primary_file_path, exist_ok=True)
        os.makedirs(temp_data_file_path, exist_ok=True)

        # Save primary file to temp path
        primary_filename = os.path.basename(urlparse(primary_file_url).path) or 'primary_file'
        base_name = os.path.splitext(primary_filename)[0]
        primary_file_path = os.path.join(temp_primary_file_path, primary_filename)

        response = requests.get(primary_file_url)
        response.raise_for_status()
        with open(primary_file_path, 'wb') as f:
            f.write(response.content)

        # Save data file to temp path with same base name as primary file
        data_extension = os.path.splitext(os.path.basename(urlparse(data_file_url).path))[1] or '.txt'
        data_filename = f'{base_name}{data_extension}'
        data_file_path = os.path.join(temp_data_file_path, data_filename)

        response = requests.get(data_file_url)
        response.raise_for_status()
        data_file_content = response.content.decode('utf-8').splitlines()

        with open(primary_file_path, 'r') as f:
            yolo_to_dm_converter = YOLOToDMConverter(temp_path, False, True, class_names)
            converted_data = yolo_to_dm_converter.convert_single_file(data_file_content, f)

        # Delete temp files
        try:
            if os.path.exists(primary_file_path):
                os.remove(primary_file_path)
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
            # Remove temp directory if empty
            if os.path.exists(temp_path) and not os.listdir(temp_path):
                os.rmdir(temp_path)
        except OSError:
            pass  # Ignore cleanup errors

        # Convert converted_data to DM Ver 1
        dm_v2_to_v1_converter = DMV2ToV1Converter(converted_data['dm_json'], 'image')
        converted_data = dm_v2_to_v1_converter.convert()
        return converted_data

    def _convert_coco(
        self,
        primary_file_url: str,
        primary_file_original_name: str,
        data_file_url: str,
        data_file_original_name: str,
        temp_path: str,
    ) -> dict:
        """Convert COCO format to DM format."""
        temp_primary_file_path = f'{temp_path}/images/'
        temp_data_file_path = f'{temp_path}/labels/'

        # Create temp_path directory if not exists
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(temp_primary_file_path, exist_ok=True)
        os.makedirs(temp_data_file_path, exist_ok=True)

        # Save primary file to temp path
        primary_filename = os.path.basename(urlparse(primary_file_url).path) or 'primary_file'
        base_name = os.path.splitext(primary_filename)[0]
        primary_file_path = os.path.join(temp_primary_file_path, primary_filename)

        response = requests.get(primary_file_url)
        response.raise_for_status()
        with open(primary_file_path, 'wb') as f:
            f.write(response.content)

        # Save data file to temp path with same base name as primary file
        data_extension = os.path.splitext(os.path.basename(urlparse(data_file_url).path))[1] or '.txt'
        data_filename = f'{base_name}{data_extension}'
        data_file_path = os.path.join(temp_data_file_path, data_filename)

        response = requests.get(data_file_url)
        response.raise_for_status()
        data_file_content = response.json()

        with open(primary_file_path, 'r') as f:
            yolo_to_dm_converter = COCOToDMConverter(temp_path, False, True)
            converted_data = yolo_to_dm_converter.convert_single_file(data_file_content, f, primary_file_original_name)

        # Delete temp files
        try:
            if os.path.exists(primary_file_path):
                os.remove(primary_file_path)
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
            # Remove temp directory if empty
            if os.path.exists(temp_path) and not os.listdir(temp_path):
                os.rmdir(temp_path)
        except OSError:
            pass  # Ignore cleanup errors

        # Convert converted_data to DM Ver 1
        dm_v2_to_v1_converter = DMV2ToV1Converter(converted_data['dm_json'], 'image')
        converted_data = dm_v2_to_v1_converter.convert()
        return converted_data

    def _convert_pascal(self, primary_file_url: str, data_file_url: str, temp_path: str) -> dict:
        """Convert Pascal format to DM format."""
        temp_primary_file_path = f'{temp_path}/images/'
        temp_data_file_path = f'{temp_path}/labels/'

        # Create temp_path directory if not exists
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(temp_primary_file_path, exist_ok=True)
        os.makedirs(temp_data_file_path, exist_ok=True)

        # Save primary file to temp path
        primary_filename = os.path.basename(urlparse(primary_file_url).path) or 'primary_file'
        base_name = os.path.splitext(primary_filename)[0]
        primary_file_path = os.path.join(temp_primary_file_path, primary_filename)

        response = requests.get(primary_file_url)
        response.raise_for_status()
        with open(primary_file_path, 'wb') as f:
            f.write(response.content)

        # Save data file to temp path with same base name as primary file
        data_extension = os.path.splitext(os.path.basename(urlparse(data_file_url).path))[1] or '.txt'
        data_filename = f'{base_name}{data_extension}'
        data_file_path = os.path.join(temp_data_file_path, data_filename)

        response = requests.get(data_file_url)
        response.raise_for_status()
        data_file_content = response.content.decode('utf-8')

        with open(primary_file_path, 'r') as f:
            yolo_to_dm_converter = PascalToDMConverter(temp_path, False, True)
            converted_data = yolo_to_dm_converter.convert_single_file(data_file_content, f)

        # Delete temp files
        try:
            if os.path.exists(primary_file_path):
                os.remove(primary_file_path)
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
            # Remove temp directory if empty
            if os.path.exists(temp_path) and not os.listdir(temp_path):
                os.rmdir(temp_path)
        except OSError:
            pass  # Ignore cleanup errors

        # Convert converted_data to DM Ver 1
        dm_v2_to_v1_converter = DMV2ToV1Converter(converted_data['dm_json'], 'image')
        converted_data = dm_v2_to_v1_converter.convert()
        return converted_data

    def _convert_dm_schema_v1(self, data_file_url: str) -> dict:
        """Load DM Schema V1 format data directly from URL (no conversion needed)."""
        response = requests.get(data_file_url)
        response.raise_for_status()
        return response.data

    def _convert_dm_schema_v2(self, data_file_url: str) -> dict:
        """Convert DM Schema V2 format to DM format."""
        response = requests.get(data_file_url)
        response.raise_for_status()
        data = response.data

        # TODO: params 를 통해 file type 받아서 처리하도록 개선
        dm_v2_to_v1_converter = DMV2ToV1Converter(data, 'image')
        converted_data = dm_v2_to_v1_converter.convert()

        return converted_data

    def convert_data_from_inference(self, data: dict) -> dict:
        """Convert the data from inference result to a task object.

        Args:
            data: Converted data.

        Returns:
            dict: The converted data.
        """
        # TODO: params 를 통해 file type 받아서 처리하도록 개선
        dm_v2_to_v1_converter = DMV2ToV1Converter(data, 'image')
        converted_data = dm_v2_to_v1_converter.convert()
        return converted_data
