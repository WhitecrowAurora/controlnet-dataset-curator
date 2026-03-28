"""
撤销/重做管理器
支持操作历史的撤销和重做功能
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """操作类型"""
    ACCEPT = "accept"
    DISCARD = "discard"
    BATCH_ACCEPT = "batch_accept"
    BATCH_DISCARD = "batch_discard"


@dataclass
class UndoAction:
    """可撤销的操作"""
    action_type: ActionType
    data: Dict[str, Any]
    undo_callback: Optional[Callable] = None
    redo_callback: Optional[Callable] = None


class UndoManager:
    """
    撤销/重做管理器

    维护操作历史栈，支持撤销和重做
    """

    def __init__(self, max_history: int = 100):
        """
        初始化

        Args:
            max_history: 最大历史记录数
        """
        self._undo_stack: List[UndoAction] = []
        self._redo_stack: List[UndoAction] = []
        self._max_history = max_history

    def push_action(self, action: UndoAction):
        """
        添加一个新操作到历史栈

        Args:
            action: 操作对象
        """
        self._undo_stack.append(action)

        # 限制历史记录数量
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

        # 清空重做栈
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        """是否可以撤销"""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """是否可以重做"""
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[UndoAction]:
        """
        撤销上一个操作

        Returns:
            被撤销的操作，如果没有可撤销的操作则返回None
        """
        if not self.can_undo():
            return None

        action = self._undo_stack.pop()
        self._redo_stack.append(action)

        # 执行撤销回调
        if action.undo_callback:
            action.undo_callback()

        return action

    def redo(self) -> Optional[UndoAction]:
        """
        重做上一个被撤销的操作

        Returns:
            被重做的操作，如果没有可重做的操作则返回None
        """
        if not self.can_redo():
            return None

        action = self._redo_stack.pop()
        self._undo_stack.append(action)

        # 执行重做回调
        if action.redo_callback:
            action.redo_callback()

        return action

    def clear(self):
        """清空所有历史记录"""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def get_undo_count(self) -> int:
        """获取可撤销操作数量"""
        return len(self._undo_stack)

    def get_redo_count(self) -> int:
        """获取可重做操作数量"""
        return len(self._redo_stack)

    def get_last_action_description(self) -> str:
        """获取最后一个操作的描述"""
        if not self.can_undo():
            return ""

        action = self._undo_stack[-1]
        action_names = {
            ActionType.ACCEPT: "接受",
            ActionType.DISCARD: "拒绝",
            ActionType.BATCH_ACCEPT: "批量接受",
            ActionType.BATCH_DISCARD: "批量拒绝",
        }
        return action_names.get(action.action_type, "未知操作")
