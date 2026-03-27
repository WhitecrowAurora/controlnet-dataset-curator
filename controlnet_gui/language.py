"""
Language Manager - Multi-language support
"""

_TRANSLATIONS = {
    'en': {
        # Settings Panel
        'data_source': 'Data Source',
        'type': 'Type',
        'local_parquet': 'Local Parquet Files',
        'streaming_dataset': 'Streaming Dataset',
        'parquet_files': 'Parquet Files',
        'add_file': 'Add File',
        'remove_file': 'Remove',
        'extract_dir': 'Extract Directory',
        'browse': 'Browse...',
        'dataset_id': 'Dataset ID or URL',
        'split': 'Split',
        'split_placeholder': 'Leave empty to auto-detect',
        'hf_token': 'HF Token',
        'hf_token_placeholder': 'Required for private repositories',
        'num_samples': 'Num Samples',
        'user_prefix': 'User Prefix',
        'user_prefix_placeholder': 'e.g., user_a (for collaboration)',
        'skip_count': 'Skip Count (Starting Position)',
        'enable_multithread': 'Enable Multi-threading',
        'thread_count': 'Threads',
        'resume_extraction': 'Resume Previous Extraction?',
        'resume_extraction_message': 'Found incomplete extraction with {} images already processed.\n\nDo you want to continue from where you left off?',
        'select_splits': 'Select Dataset Splits',
        'available_splits_info': 'The following splits are available in this dataset. Please select which ones to extract:',
        'no_splits_found': 'No Splits Found',
        'no_splits_found_message': 'No splits found in this dataset.',
        'no_split_selected': 'No Split Selected',
        'no_split_selected_message': 'Please select at least one split to extract.',
        'multiple_splits_info': 'Multiple Splits Selected',
        'multiple_splits_message': 'Multiple splits selected. Currently only extracting: {}',
        'error': 'Error',
        'datasets_not_installed': 'datasets library not installed. Please run: pip install datasets',
        'failed_to_detect_splits': 'Failed to detect splits',
        'extraction_failed': 'Extraction failed',
        'auto_detect_failed': 'Auto-detect Failed',
        'auto_detect_failed_message': 'Failed to auto-detect splits (torch DLL error). Use default split "train"?',
        'use_default_split': 'Use default split "train"?',
        'torch_dll_error_title': 'Cannot auto-detect splits due to torch DLL error',
        'torch_dll_error': 'Torch DLL Loading Error',
        'torch_dll_error_datasets': 'Failed to load datasets library due to torch DLL error',
        'torch_dll_error_datasets_info': 'The datasets library (version 4.x) requires torch, but torch DLL failed to load in GUI environment.\n\nClick "Fix Now" to downgrade datasets to version 2.x, which works without torch.',
        'fix_now': 'Fix Now (Downgrade Datasets)',
        'torch_dll_error_message': 'Auto-detection requires torch library, but DLL loading failed.\n\nYou can:\n1. Use default split "train" (recommended)\n2. Manually enter split name in the field above\n\nNote: This only affects split detection. Extraction will still work.',
        'use_train_split': 'Use "train"',
        'cancel': 'Cancel',
        'extract_all': 'Extract All (0 = all)',

        'processing': 'Processing',
        'input_source': 'Input Source',
        'use_data_source_dir': 'Use Data Source Directory',
        'use_custom_dir': 'Use Custom Directory',
        'threads': 'Threads',
        'preload_count': 'Preload Count',
        'control_types': 'Control Types',
        'model': 'Model',
        'model_path_placeholder': 'Enter model path or HuggingFace model ID',
        'quality_profile': 'Quality Profile',
        'general': 'General',
        'anime': 'Anime',
        'enable_retry': 'Enable Smart Retry',
        'max_retries': 'Max Retries',

        'output': 'Output',
        'output_dir': 'Output Dir',
        'discard_action': 'Discard Action',
        'move_to_trash': 'Move to Trash',
        'delete_permanently': 'Delete Permanently',

        'custom_tags': 'Custom Tags',
        'append_tags': 'Append Tags',
        'tag_placeholder': 'quality, detailed, clean lineart',

        'start_processing': 'Start Processing',
        'pause_processing': 'Pause Processing',
        'resume_processing': 'Resume Processing',
        'stop_processing': 'Stop Processing',
        'extract_data': 'Extract Data',
        'detecting_splits': 'Detecting Splits...',

        # Menu
        'file': 'File',
        'edit': 'Edit',
        'help': 'Help',
        'load_config': 'Load Config...',
        'save_config': 'Save Config',
        'exit': 'Exit',
        'clear_all_images': 'Clear All Images',
        'about': 'About',
        'language': 'Language',
        'delete_behavior': 'Delete Behavior',
        'delete_permanent': 'Delete Permanently',
        'delete_7days': 'Keep 7 Days',
        'delete_30days': 'Keep 30 Days',
        'delete_never': 'Never Delete',
        'cleanup_delete_folder': 'Cleanup .delete Folder',
        'delete_never_no_cleanup': 'Delete behavior is set to "Never Delete", no cleanup needed.',
        'delete_folder_empty': '.delete folder is empty or does not exist.',
        'no_files_to_cleanup': 'No files to cleanup based on retention policy.',
        'confirm_cleanup': 'Confirm cleanup',
        'files': 'files',
        'retention_policy': 'Retention policy',
        'days': 'days',
        'cleanup_complete': 'Cleanup complete',

        # Tools menu
        'tools': 'Tools',
        'install_torch': 'Install PyTorch',
        'reinstall_torch': 'Reinstall PyTorch',
        'install_datasets': 'Install Datasets Library',
        'fix_datasets': 'Fix Datasets Library (Downgrade)',
        'install_all_dependencies': 'Install All Dependencies',
        'check_dependencies': 'Check Dependencies',
        'install_torch_confirm': 'This will install PyTorch and torchvision.\n\nThis may take several minutes and requires internet connection.\n\nContinue?',
        'reinstall_torch_confirm': 'This will reinstall PyTorch and torchvision (force reinstall).\n\nThis may take several minutes and requires internet connection.\n\nContinue?',
        'reinstalling_torch': 'Reinstalling PyTorch',
        'install_datasets_confirm': 'This will install the datasets library.\n\nThis may take a few minutes and requires internet connection.\n\nContinue?',
        'fix_datasets_confirm': 'This will downgrade datasets library to version 2.x.\n\nVersion 2.x works without torch in GUI environment.\nThis fixes the "torch DLL loading failed" error.\n\nContinue?',
        'fixing_datasets': 'Fixing Datasets Library',
        'install_all_dependencies_confirm': 'This will install all required dependencies:\n- PyTorch\n- torchvision\n- datasets\n- controlnet-aux\n- opencv-python\n\nThis may take 10-20 minutes and requires internet connection.\n\nContinue?',
        'installing_torch': 'Installing PyTorch',
        'installing_datasets': 'Installing Datasets',
        'installing_all_dependencies': 'Installing All Dependencies',
        'installing': 'Installing',
        'restart_now': 'Installation complete! Restart application now?',
        'please_wait': 'Please wait, this may take several minutes...',
        'close': 'Close',
        'check_console_output': 'Please check the console output for details.',
        'installation_complete': 'Installation Complete',
        'successfully_installed': 'Successfully installed',
        'restart_recommended': 'Please restart the application for changes to take effect.',
        'installation_failed': 'Installation Failed',
        'failed_to_install': 'Failed to install',
        'installation_error': 'Installation error',
        'dependencies_status': 'Dependencies Status:',
        'not_installed': 'Not installed',
        'error_loading': 'Error loading',

        # Image List
        'no_data': 'No data\nPlease configure data source and click "Start Processing"',
        'original': 'Original',
        'confirm': '✓ Confirm',
        'discard': '✗ Discard',

        # Preview Panel
        'preview': 'Preview',
        'no_selection': 'No image selected\nClick on an image to preview',
        'score': 'Score',
        'total_score': 'Total Score',
        'quality_level': 'Quality Level',
        'canny_score': 'Canny Score',
        'white_ratio': 'White Ratio',
        'noise_count': 'Noise Count',
        'avg_thickness': 'Avg Thickness',
        'openpose_score': 'OpenPose Score',
        'depth_score': 'Depth Score',
        'auto_accept': 'Auto Accept',
        'auto_reject': 'Auto Reject',
        'need_review': 'Need Review',

        # Status messages
        'processing_started': 'Processing started, running continuously...',
        'processing_paused': 'Processing paused',
        'processing_resumed': 'Processing resumed',
        'processing_complete': 'Processing complete',
        'extracting_data': 'Extracting data from parquet files...',
        'extraction_complete': 'Extraction complete',
        'extracted_count': 'Extracted',
        'saved': 'Saved',
        'discarded': 'Discarded',
        'images_remaining': 'images remaining',
        'images_displaying': 'images displaying',
        'auto_accepted': 'Auto Accepted',
        'auto_rejected': 'Auto Rejected',
        'need_review': 'Need Review',
        'processed': 'Processed',
        'in_queue': 'In Queue',
        'inbox_pending': 'Inbox Pending',
        'displaying': 'Displaying',
        'total': 'Total',
        'all_images_processed': 'All images processed!',
        'save_failed': 'Save Failed',
        'cannot_save_image': 'Cannot save image',
        'advanced': 'Advanced Settings',
        'all': 'All',
        'select_files': 'Select Files',

        # Quality warnings
        'quality_warning': 'Quality Warning',
        'low_sharpness': 'Low Sharpness (Blur)',
        'medium_sharpness': 'Medium Sharpness',
        'high_sharpness': 'High Sharpness',
        'no_pose_detected': 'No Pose Detected',
        'low_depth_quality': 'Low Depth Quality',
        'retry_count': 'Retry',

        # Dialogs
        'select_directory': 'Select Directory',
        'confirm_clear': 'Are you sure you want to clear all images?',
        'confirm_exit': 'Processing is still running. Are you sure you want to exit?',
        'yes': 'Yes',
        'no': 'No',

        # About
        'about_title': 'About',
        'about_text': '''ControlNet Data Processing Tool

A portable tool for processing and filtering ControlNet training data.

Features:
- Multi-threshold Canny edge detection
- OpenPose and Depth map generation
- Automatic quality scoring
- Three-tier filtering system
- Streaming and local file support'''
    },
    'zh': {
        # Settings Panel
        'data_source': '数据源',
        'type': '类型',
        'local_parquet': '本地Parquet文件',
        'streaming_dataset': '流式数据集',
        'parquet_files': 'Parquet文件',
        'add_file': '添加文件',
        'remove_file': '移除',
        'extract_dir': '提取目录',
        'browse': '浏览...',
        'dataset_id': '数据集ID或链接',
        'split': '分割',
        'split_placeholder': '留空自动探测',
        'hf_token': 'HF令牌',
        'hf_token_placeholder': '私有仓库需添加',
        'num_samples': '样本数量',
        'user_prefix': '用户前缀',
        'user_prefix_placeholder': '例如: user_a (用于协作)',
        'skip_count': '跳过数量（起始位置）',
        'enable_multithread': '启用多线程',
        'thread_count': '线程数',
        'resume_extraction': '继续上次提取？',
        'resume_extraction_message': '发现未完成的提取任务，已处理 {} 张图片。\n\n是否从上次中断的地方继续？',
        'select_splits': '选择数据集分割',
        'available_splits_info': '此数据集包含以下分割，请选择要提取的分割：',
        'no_splits_found': '未找到分割',
        'no_splits_found_message': '此数据集中未找到任何分割。',
        'no_split_selected': '未选择分割',
        'no_split_selected_message': '请至少选择一个要提取的分割。',
        'multiple_splits_info': '选择了多个分割',
        'multiple_splits_message': '已选择多个分割，当前仅提取：{}',
        'error': '错误',
        'datasets_not_installed': '未安装 datasets 库，请运行：pip install datasets',
        'failed_to_detect_splits': '探测分割失败',
        'extraction_failed': '提取失败',
        'auto_detect_failed': '自动探测失败',
        'auto_detect_failed_message': '自动探测分割失败（torch DLL错误）。使用默认分割 "train"？',
        'use_default_split': '使用默认分割 "train"？',
        'torch_dll_error_title': '由于 torch DLL 错误无法自动探测分割',
        'torch_dll_error': 'Torch DLL 加载错误',
        'torch_dll_error_datasets': '由于 torch DLL 错误导致 datasets 库加载失败',
        'torch_dll_error_datasets_info': 'datasets 库（4.x 版本）需要 torch，但 torch DLL 在 GUI 环境下加载失败。\n\n点击"立即修复"将 datasets 降级到 2.x 版本，该版本可以不依赖 torch 工作。',
        'fix_now': '立即修复（降级 Datasets）',
        'torch_dll_error_message': '自动探测需要 torch 库，但 DLL 加载失败。\n\n您可以：\n1. 使用默认分割 "train"（推荐）\n2. 在上方字段手动输入分割名称\n\n注意：这只影响分割探测，提取功能仍然可用。',
        'use_train_split': '使用 "train"',
        'cancel': '取消',
        'extract_all': '提取全部 (0 = 全部)',

        'processing': '处理设置',
        'input_source': '输入来源',
        'use_data_source_dir': '使用数据源目录',
        'use_custom_dir': '使用自定义目录',
        'threads': '线程数',
        'preload_count': '预加载数量',
        'control_types': '控制类型',
        'model': '模型',
        'model_path_placeholder': '输入模型路径或 HuggingFace 模型 ID',
        'quality_profile': '质量配置',
        'general': '通用',
        'anime': '动漫',
        'enable_retry': '启用智能重试',
        'max_retries': '最大重试次数',

        'output': '输出设置',
        'output_dir': '输出目录',
        'discard_action': '废弃操作',
        'move_to_trash': '移动到废弃文件夹',
        'delete_permanently': '彻底删除',

        'custom_tags': '自定义标签',
        'append_tags': '追加标签',
        'tag_placeholder': 'quality, detailed, clean lineart',

        'start_processing': '开始处理',
        'pause_processing': '暂停处理',
        'resume_processing': '继续处理',
        'stop_processing': '停止处理',
        'extract_data': '提取数据',
        'detecting_splits': '正在检测分割...',

        # Menu
        'file': '文件',
        'edit': '编辑',
        'help': '帮助',
        'load_config': '加载配置...',
        'save_config': '保存配置',
        'exit': '退出',
        'clear_all_images': '清空所有图片',
        'about': '关于',
        'language': '语言',
        'delete_behavior': '删除行为',
        'delete_permanent': '彻底删除',
        'delete_7days': '保留7天',
        'delete_30days': '保留30天',
        'delete_never': '永不删除',
        'cleanup_delete_folder': '清理.delete文件夹',
        'delete_never_no_cleanup': '删除行为设置为"永不删除"，无需清理。',
        'delete_folder_empty': '.delete文件夹为空或不存在。',
        'no_files_to_cleanup': '根据保留策略，没有需要清理的文件。',
        'confirm_cleanup': '确认清理',
        'files': '个文件',
        'retention_policy': '保留策略',
        'days': '天',
        'cleanup_complete': '清理完成',

        # Tools menu
        'tools': '工具',
        'install_torch': '安装 PyTorch',
        'reinstall_torch': '重装 PyTorch',
        'install_datasets': '安装 Datasets 库',
        'fix_datasets': '修复 Datasets 库（降级）',
        'install_all_dependencies': '安装所有依赖',
        'check_dependencies': '检查依赖',
        'install_torch_confirm': '这将安装 PyTorch 和 torchvision。\n\n可能需要几分钟时间，需要网络连接。\n\n是否继续？',
        'reinstall_torch_confirm': '这将重新安装 PyTorch 和 torchvision（强制重装）。\n\n可能需要几分钟时间，需要网络连接。\n\n是否继续？',
        'reinstalling_torch': '正在重装 PyTorch',
        'install_datasets_confirm': '这将安装 datasets 库。\n\n可能需要几分钟时间，需要网络连接。\n\n是否继续？',
        'fix_datasets_confirm': '这将降级 datasets 库到 2.x 版本。\n\n2.x 版本在 GUI 环境下可以不依赖 torch 工作。\n这将修复"torch DLL 加载失败"错误。\n\n是否继续？',
        'fixing_datasets': '正在修复 Datasets 库',
        'install_all_dependencies_confirm': '这将安装所有必需的依赖：\n- PyTorch\n- torchvision\n- datasets\n- controlnet-aux\n- opencv-python\n\n可能需要 10-20 分钟，需要网络连接。\n\n是否继续？',
        'installing_torch': '正在安装 PyTorch',
        'installing_datasets': '正在安装 Datasets',
        'installing_all_dependencies': '正在安装所有依赖',
        'installing': '正在安装',
        'please_wait': '请稍候，这可能需要几分钟...',
        'close': '关闭',
        'check_console_output': '请查看控制台输出了解详情。',
        'installation_complete': '安装完成',
        'successfully_installed': '成功安装',
        'restart_now': '安装完成！是否立即重启应用程序？',
        'restart_recommended': '请重启应用程序以使更改生效。',
        'installation_failed': '安装失败',
        'failed_to_install': '安装失败',
        'installation_error': '安装错误',
        'dependencies_status': '依赖状态：',
        'not_installed': '未安装',
        'error_loading': '加载错误',

        # Image List
        'no_data': '暂无数据\n请配置数据源并点击"开始处理"',
        'original': '原图',
        'confirm': '✓ 确认',
        'discard': '✗ 废弃',

        # Preview Panel
        'preview': '预览',
        'no_selection': '未选择图片\n点击图片进行预览',
        'score': '分数',
        'total_score': '综合分数',
        'quality_level': '质量等级',
        'canny_score': 'Canny分数',
        'white_ratio': '白色占比',
        'noise_count': '噪点数量',
        'avg_thickness': '平均粗细',
        'openpose_score': 'OpenPose分数',
        'depth_score': 'Depth分数',
        'auto_accept': '自动通过',
        'auto_reject': '自动废弃',
        'need_review': '需要审核',

        # Status messages
        'processing_started': '处理已启动，持续运行中...',
        'processing_paused': '处理已暂停',
        'processing_resumed': '处理已恢复',
        'processing_complete': '处理完成',
        'extracting_data': '正在从parquet文件提取数据...',
        'extraction_complete': '提取完成',
        'extracted_count': '已提取',
        'saved': '已保存',
        'discarded': '已废弃',
        'images_remaining': '张剩余',
        'images_displaying': '张显示中',
        'auto_accepted': '自动通过',
        'auto_rejected': '自动废弃',
        'need_review': '需要审核',
        'processed': '已处理',
        'in_queue': '队列中',
        'inbox_pending': '审核箱待处理',
        'displaying': '显示中',
        'total': '总计',
        'all_images_processed': '所有图片处理完成！',
        'save_failed': '保存失败',
        'cannot_save_image': '无法保存图片',
        'advanced': '高级设置',
        'all': '全部',
        'select_files': '选择文件',

        # Quality warnings
        'quality_warning': '质量警告',
        'low_sharpness': '清晰度低（模糊）',
        'medium_sharpness': '清晰度中等',
        'high_sharpness': '清晰度高',
        'no_pose_detected': '未检测到姿态',
        'low_depth_quality': '深度质量低',
        'retry_count': '重试',

        # Dialogs
        'select_directory': '选择目录',
        'confirm_clear': '确定要清空所有图片吗？',
        'confirm_exit': '处理仍在运行中，确定要退出吗？',
        'yes': '是',
        'no': '否',

        # About
        'about_title': '关于',
        'about_text': '''ControlNet 数据处理工具

用于处理和筛选ControlNet训练数据的便携式工具。

功能特性：
- 多阈值Canny边缘检测
- OpenPose和Depth图生成
- 自动质量评分
- 三层筛选系统
- 支持流式和本地文件'''
    }
}

class LanguageManager:
    """Manage UI language translations"""

    def __init__(self, language='en'):
        self.language = language
        self.translations = {lang: dict(values) for lang, values in _TRANSLATIONS.items()}

    def set_language(self, language: str):
        """Set current language"""
        if language in self.translations:
            self.language = language

    def get(self, key: str, default: str = '') -> str:
        """Get translation for key"""
        return self.translations.get(self.language, {}).get(key, default or key)

    def __call__(self, key: str, default: str = '') -> str:
        """Shorthand for get()"""
        return self.get(key, default)


# Global language manager instance
_lang_manager = LanguageManager('zh')  # Default to Chinese

def get_lang_manager() -> LanguageManager:
    """Get global language manager instance"""
    return _lang_manager

def tr(key: str, default: str = '') -> str:
    """Translate key - shorthand function"""
    return _lang_manager.get(key, default)