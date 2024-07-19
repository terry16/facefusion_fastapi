import os
import uuid
import requests
import shutil
import signal
import onnxruntime
import numpy
from time import time, sleep
from argparse import ArgumentParser, HelpFormatter
import facefusion.choices
import facefusion.globals
from facefusion.execution import encode_execution_providers, decode_execution_providers
from facefusion.core import process_image, destroy, apply_config, apply_args, validate_args, conditional_process
from facefusion import face_analyser, face_masker, content_analyser, config, process_manager, metadata, logger, wording, voice_extractor
from facefusion.common_helper import create_metavar, get_first
from facefusion.filesystem import get_temp_frame_paths, get_temp_file_path, create_temp, move_temp, clear_temp, is_image, is_video, filter_audio_paths, resolve_relative_path, list_directory
from facefusion.processors.frame.core import get_frame_processors_modules, load_frame_processor_module

def download_image(url, download_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
    else:
        raise Exception(f"Failed to download image from {url}")

def initialize_parameters():
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = ArgumentParser(formatter_class=lambda prog: HelpFormatter(prog, max_help_position=200), add_help=False)
    # general
    program.add_argument('-c', '--config', help=wording.get('help.config'), dest='config_path', default='facefusion.ini')
    program.add_argument('-s', '--source', help=wording.get('help.source'), action='append', dest='source_paths', default=facefusion.globals.source_paths or config.get_str_list('general.source_paths'))
    program.add_argument('-t', '--target', help=wording.get('help.target'), dest='target_path', default=facefusion.globals.target_path or config.get_str_value('general.target_path'))
    program.add_argument('-o', '--output', help=wording.get('help.output'), dest='output_path', default=facefusion.globals.output_path or config.get_str_value('general.output_path'))
    program.add_argument('-v', '--version', version=metadata.get('name') + ' ' + metadata.get('version'), action='version')
    apply_config(program)
    # misc
    group_misc = program.add_argument_group('misc')
    group_misc.add_argument('--force-download', help = wording.get('help.force_download'), action = 'store_true', default = config.get_bool_value('misc.force_download'))
    group_misc.add_argument('--skip-download', help = wording.get('help.skip_download'), action = 'store_true', default = config.get_bool_value('misc.skip_download'))
    group_misc.add_argument('--headless', help = wording.get('help.headless'), action = 'store_true', default = config.get_bool_value('misc.headless'))
    group_misc.add_argument('--log-level', help = wording.get('help.log_level'), default = config.get_str_value('misc.log_level', 'info'), choices = logger.get_log_levels())
    # execution
    execution_providers = encode_execution_providers(onnxruntime.get_available_providers())
    group_execution = program.add_argument_group('execution')
    group_execution.add_argument('--execution-device-id', help = wording.get('help.execution_device_id'), default = config.get_str_value('execution.face_detector_size', '0'))
    group_execution.add_argument('--execution-providers', help = wording.get('help.execution_providers').format(choices = ', '.join(execution_providers)), default = config.get_str_list('execution.execution_providers', 'cpu'), choices = execution_providers, nargs = '+', metavar = 'EXECUTION_PROVIDERS')
    group_execution.add_argument('--execution-thread-count', help = wording.get('help.execution_thread_count'), type = int, default = config.get_int_value('execution.execution_thread_count', '4'), choices = facefusion.choices.execution_thread_count_range, metavar = create_metavar(facefusion.choices.execution_thread_count_range))
    group_execution.add_argument('--execution-queue-count', help = wording.get('help.execution_queue_count'), type = int, default = config.get_int_value('execution.execution_queue_count', '1'), choices = facefusion.choices.execution_queue_count_range, metavar = create_metavar(facefusion.choices.execution_queue_count_range))
    # memory
    group_memory = program.add_argument_group('memory')
    group_memory.add_argument('--video-memory-strategy', help = wording.get('help.video_memory_strategy'), default = config.get_str_value('memory.video_memory_strategy', 'strict'), choices = facefusion.choices.video_memory_strategies)
    group_memory.add_argument('--system-memory-limit', help = wording.get('help.system_memory_limit'), type = int, default = config.get_int_value('memory.system_memory_limit', '0'), choices = facefusion.choices.system_memory_limit_range, metavar = create_metavar(facefusion.choices.system_memory_limit_range))
    # face analyser
    group_face_analyser = program.add_argument_group('face analyser')
    group_face_analyser.add_argument('--face-analyser-order', help = wording.get('help.face_analyser_order'), default = config.get_str_value('face_analyser.face_analyser_order', 'left-right'), choices = facefusion.choices.face_analyser_orders)
    group_face_analyser.add_argument('--face-analyser-age', help = wording.get('help.face_analyser_age'), default = config.get_str_value('face_analyser.face_analyser_age'), choices = facefusion.choices.face_analyser_ages)
    group_face_analyser.add_argument('--face-analyser-gender', help = wording.get('help.face_analyser_gender'), default = config.get_str_value('face_analyser.face_analyser_gender'), choices = facefusion.choices.face_analyser_genders)
    group_face_analyser.add_argument('--face-detector-model', help = wording.get('help.face_detector_model'), default = config.get_str_value('face_analyser.face_detector_model', 'yoloface'), choices = facefusion.choices.face_detector_set.keys())
    group_face_analyser.add_argument('--face-detector-size', help = wording.get('help.face_detector_size'), default = config.get_str_value('face_analyser.face_detector_size', '640x640'))
    group_face_analyser.add_argument('--face-detector-score', help = wording.get('help.face_detector_score'), type = float, default = config.get_float_value('face_analyser.face_detector_score', '0.5'), choices = facefusion.choices.face_detector_score_range, metavar = create_metavar(facefusion.choices.face_detector_score_range))
    group_face_analyser.add_argument('--face-landmarker-score', help = wording.get('help.face_landmarker_score'), type = float, default = config.get_float_value('face_analyser.face_landmarker_score', '0.5'), choices = facefusion.choices.face_landmarker_score_range, metavar = create_metavar(facefusion.choices.face_landmarker_score_range))
    # face selector
    group_face_selector = program.add_argument_group('face selector')
    group_face_selector.add_argument('--face-selector-mode', help = wording.get('help.face_selector_mode'), default = config.get_str_value('face_selector.face_selector_mode', 'reference'), choices = facefusion.choices.face_selector_modes)
    group_face_selector.add_argument('--reference-face-position', help = wording.get('help.reference_face_position'), type = int, default = config.get_int_value('face_selector.reference_face_position', '0'))
    group_face_selector.add_argument('--reference-face-distance', help = wording.get('help.reference_face_distance'), type = float, default = config.get_float_value('face_selector.reference_face_distance', '0.6'), choices = facefusion.choices.reference_face_distance_range, metavar = create_metavar(facefusion.choices.reference_face_distance_range))
    group_face_selector.add_argument('--reference-frame-number', help = wording.get('help.reference_frame_number'), type = int, default = config.get_int_value('face_selector.reference_frame_number', '0'))
    # face mask
    group_face_mask = program.add_argument_group('face mask')
    group_face_mask.add_argument('--face-mask-types', help = wording.get('help.face_mask_types').format(choices = ', '.join(facefusion.choices.face_mask_types)), default = config.get_str_list('face_mask.face_mask_types', 'box'), choices = facefusion.choices.face_mask_types, nargs = '+', metavar = 'FACE_MASK_TYPES')
    group_face_mask.add_argument('--face-mask-blur', help = wording.get('help.face_mask_blur'), type = float, default = config.get_float_value('face_mask.face_mask_blur', '0.3'), choices = facefusion.choices.face_mask_blur_range, metavar = create_metavar(facefusion.choices.face_mask_blur_range))
    group_face_mask.add_argument('--face-mask-padding', help = wording.get('help.face_mask_padding'), type = int, default = config.get_int_list('face_mask.face_mask_padding', '0 0 0 0'), nargs = '+')
    group_face_mask.add_argument('--face-mask-regions', help = wording.get('help.face_mask_regions').format(choices = ', '.join(facefusion.choices.face_mask_regions)), default = config.get_str_list('face_mask.face_mask_regions', ' '.join(facefusion.choices.face_mask_regions)), choices = facefusion.choices.face_mask_regions, nargs = '+', metavar = 'FACE_MASK_REGIONS')
    # frame extraction
    group_frame_extraction = program.add_argument_group('frame extraction')
    group_frame_extraction.add_argument('--trim-frame-start', help = wording.get('help.trim_frame_start'), type = int, default = facefusion.config.get_int_value('frame_extraction.trim_frame_start'))
    group_frame_extraction.add_argument('--trim-frame-end',	help = wording.get('help.trim_frame_end'), type = int, default = facefusion.config.get_int_value('frame_extraction.trim_frame_end'))
    group_frame_extraction.add_argument('--temp-frame-format', help = wording.get('help.temp_frame_format'), default = config.get_str_value('frame_extraction.temp_frame_format', 'png'), choices = facefusion.choices.temp_frame_formats)
    group_frame_extraction.add_argument('--keep-temp', help = wording.get('help.keep_temp'), action = 'store_true',	default = config.get_bool_value('frame_extraction.keep_temp'))
    # output creation
    group_output_creation = program.add_argument_group('output creation')
    group_output_creation.add_argument('--output-image-quality', help = wording.get('help.output_image_quality'), type = int, default = config.get_int_value('output_creation.output_image_quality', '80'), choices = facefusion.choices.output_image_quality_range, metavar = create_metavar(facefusion.choices.output_image_quality_range))
    group_output_creation.add_argument('--output-image-resolution', help = wording.get('help.output_image_resolution'), default = config.get_str_value('output_creation.output_image_resolution'))
    group_output_creation.add_argument('--output-video-encoder', help = wording.get('help.output_video_encoder'), default = config.get_str_value('output_creation.output_video_encoder', 'libx264'), choices = facefusion.choices.output_video_encoders)
    group_output_creation.add_argument('--output-video-preset', help = wording.get('help.output_video_preset'), default = config.get_str_value('output_creation.output_video_preset', 'veryfast'), choices = facefusion.choices.output_video_presets)
    group_output_creation.add_argument('--output-video-quality', help = wording.get('help.output_video_quality'), type = int, default = config.get_int_value('output_creation.output_video_quality', '80'), choices = facefusion.choices.output_video_quality_range, metavar = create_metavar(facefusion.choices.output_video_quality_range))
    group_output_creation.add_argument('--output-video-resolution', help = wording.get('help.output_video_resolution'), default = config.get_str_value('output_creation.output_video_resolution'))
    group_output_creation.add_argument('--output-video-fps', help = wording.get('help.output_video_fps'), type = float, default = config.get_str_value('output_creation.output_video_fps'))
    group_output_creation.add_argument('--skip-audio', help = wording.get('help.skip_audio'), action = 'store_true', default = config.get_bool_value('output_creation.skip_audio'))
    # frame processors
    available_frame_processors = list_directory('facefusion/processors/frame/modules')
    program = ArgumentParser(parents = [ program ], formatter_class = program.formatter_class, add_help = True)
    group_frame_processors = program.add_argument_group('frame processors')
    group_frame_processors.add_argument('--frame-processors', help = wording.get('help.frame_processors').format(choices = ', '.join(available_frame_processors)), default = config.get_str_list('frame_processors.frame_processors', 'face_swapper'), nargs = '+')
    for frame_processor in available_frame_processors:
        frame_processor_module = load_frame_processor_module(frame_processor)
        frame_processor_module.register_args(group_frame_processors)
    # uis
    available_ui_layouts = list_directory('facefusion/uis/layouts')
    group_uis = program.add_argument_group('uis')
    group_uis.add_argument('--open-browser', help=wording.get('help.open_browser'), action = 'store_true', default = config.get_bool_value('uis.open_browser'))
    group_uis.add_argument('--ui-layouts', help = wording.get('help.ui_layouts').format(choices = ', '.join(available_ui_layouts)), default = config.get_str_list('uis.ui_layouts', 'default'), nargs = '+')
    return program

def pre_check() -> bool:
    if sys.version_info < (3, 9):
        logger.error(wording.get('python_not_supported').format(version='3.9'), __name__.upper())
        return False
    if not shutil.which('ffmpeg'):
        logger.error(wording.get('ffmpeg_not_installed'), __name__.upper())
        return False
    return True

def process_face_swap(srcUrl, targetUrl):
    # 下载图片
    src_path = os.path.join('input', f"{uuid.uuid4()}.jpg")
    target_path = os.path.join('input', f"{uuid.uuid4()}.jpg")
    download_image(srcUrl, src_path)
    download_image(targetUrl, target_path)
    
    # 设置全局参数
    facefusion.globals.source_paths = [src_path]
    facefusion.globals.target_path = target_path
    facefusion.globals.output_path = os.path.join('output', f"{uuid.uuid4()}.jpg")

    # 初始化参数解析器
    program = initialize_parameters()

    # 验证和应用参数
    validate_args(program)
    apply_args(program)
    
    # 处理图片
    conditional_process();

    # 返回输出路径
    return facefusion.globals.output_path