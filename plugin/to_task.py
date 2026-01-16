import copy
import os
import re
from collections import defaultdict
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
            yolo_class_names = yolo_class_names.replace(' ', '').split(',')
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
            converted_data = self._convert_dm_schema_v2(data_file_url)
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
        dm_v2_to_v1_converter = DMV2ToV1Converter()
        converted_data = dm_v2_to_v1_converter.convert(converted_data['dm_json'])
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
        try:
            json_data = response.json()
            converted_data = self._normalize_dm_schema_v1_nested_dictionary_keys(json_data)
            return converted_data
        except ValueError as e:
            raise ValueError(f'Failed to parse JSON response: {str(e)}')

    def _convert_dm_schema_v2(self, data_file_url: str) -> dict:
        """Convert DM Schema V2 format to DM format."""
        response = requests.get(data_file_url)
        response.raise_for_status()
        json_data = response.json()

        # TODO: params 를 통해 file type 받아서 처리하도록 개선
        dm_v2_to_v1_converter = DMV2ToV1Converter(json_data, 'image')
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

    def _normalize_dm_schema_v1_nested_dictionary_keys(self, data):
        """
        특정 dictionary의 중첩된 구조에서 "_number" 형태가 없는 키에 번호를 붙여주는 메서드

        처리 대상: extra, relations, annotations, annotationsData, annotationGroups
        각 dict 내부의 키들에 대해 "_number" 형태가 없으면 추가

        Args:
            data (dict): 처리할 데이터 딕셔너리

        Returns:
            dict: 키가 정규화된 새로운 딕셔너리
        """
        # 처리 대상 키 목록
        target_keys = {'extra', 'relations', 'annotations', 'annotationsData', 'annotationGroups'}

        # 깊은 복사로 원본 보존
        normalized_data = copy.deepcopy(data)

        for main_key in target_keys:
            if main_key in normalized_data and isinstance(normalized_data[main_key], dict):
                # 해당 딕셔너리의 키들을 정규화
                normalized_data[main_key] = self._normalize_dm_schema_v1_keys_with_incremental_numbers(
                    normalized_data[main_key]
                )

        return normalized_data

    def _normalize_dm_schema_v1_keys_with_incremental_numbers(self, nested_dict):
        """
        딕셔너리 내부의 키들에 순차적으로 번호를 붙이는 함수
        같은 기본 키가 여러 개 있으면 _1, _2, _3 순으로 번호 증가

        Args:
            nested_dict (dict): 정규화할 중첩 딕셔너리

        Returns:
            dict: 키가 정규화된 딕셔너리
        """
        if not isinstance(nested_dict, dict):
            return nested_dict

        # 기본 키별 카운터
        key_counters = defaultdict(int)
        normalized_dict = {}

        # 먼저 모든 키를 순회하여 기본 키 추출 및 카운팅
        base_key_groups = defaultdict(list)

        for key in nested_dict.keys():
            # "_숫자" 패턴이 있으면 기본 키 추출, 없으면 전체가 기본 키
            base_key = re.sub(r'_\d+$', '', key)
            base_key_groups[base_key].append(key)

        # 각 기본 키 그룹별로 순차적으로 번호 할당
        for base_key, original_keys in base_key_groups.items():
            for i, original_key in enumerate(original_keys, 1):
                new_key = f'{base_key}_{i}'
                normalized_dict[new_key] = nested_dict[original_key]

        return normalized_dict

    def analyze_dm_schema_v1_dictionary_structure(self, data):
        """
        딕셔너리 구조를 분석하여 정규화가 필요한 부분을 식별하는 함수

        Args:
            data (dict): 분석할 데이터

        Returns:
            dict: 구조 분석 결과
        """
        target_keys = {'extra', 'relations', 'annotations', 'annotationsData', 'annotationGroups'}
        analysis = {'target_sections': {}, 'needs_normalization': {}, 'summary': {}}

        for main_key in target_keys:
            if main_key in data and isinstance(data[main_key], dict):
                section_info = {
                    'total_keys': len(data[main_key]),
                    'keys_with_number': [],
                    'keys_without_number': [],
                    'base_key_counts': defaultdict(int),
                }

                for key in data[main_key].keys():
                    if re.search(r'_\d+$', key):
                        section_info['keys_with_number'].append(key)
                    else:
                        section_info['keys_without_number'].append(key)

                    # 기본 키 카운팅
                    base_key = re.sub(r'_\d+$', '', key)
                    section_info['base_key_counts'][base_key] += 1

                analysis['target_sections'][main_key] = section_info
                analysis['needs_normalization'][main_key] = len(section_info['keys_without_number']) > 0 or any(
                    count > 1 for count in section_info['base_key_counts'].values()
                )

        # 전체 요약
        total_sections = len(analysis['target_sections'])
        sections_needing_norm = sum(analysis['needs_normalization'].values())

        analysis['summary'] = {
            'total_target_sections': total_sections,
            'sections_needing_normalization': sections_needing_norm,
            'normalization_required': sections_needing_norm > 0,
        }

        return analysis

    def compare_normalized_dm_schema_v1_before_after(self, original_data, normalized_data):
        """
        정규화 전후 비교 함수

        Args:
            original_data (dict): 원본 데이터
            normalized_data (dict): 정규화된 데이터

        Returns:
            dict: 비교 결과
        """
        target_keys = {'extra', 'relations', 'annotations', 'annotationsData', 'annotationGroups'}
        comparison = {}

        for main_key in target_keys:
            if main_key in original_data and main_key in normalized_data:
                original_keys = set(original_data[main_key].keys())
                normalized_keys = set(normalized_data[main_key].keys())

                comparison[main_key] = {
                    'original_keys': sorted(original_keys),
                    'normalized_keys': sorted(normalized_keys),
                    'key_mappings': [],
                }

                # 키 매핑 생성 (순서 기반)
                orig_list = list(original_data[main_key].keys())
                norm_list = list(normalized_data[main_key].keys())

                for orig_key, norm_key in zip(orig_list, norm_list):
                    comparison[main_key]['key_mappings'].append({
                        'original': orig_key,
                        'normalized': norm_key,
                        'changed': orig_key != norm_key,
                    })

        return comparison
