"""
Модели данных для парсера xdic.
Датаклассы для представления структуры БД.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FieldInfo:
    """Описание одного поля (колонки) таблицы."""
    name: str
    field_type: str  # Тип из xdic: Текст, Строка, Целое, Дата, Внешний ключ и т.д.
    description: Optional[str] = None
    length: Optional[int] = None
    precision: Optional[int] = None
    size: Optional[int] = None
    default_value: Optional[str] = None
    referenced_table: Optional[str] = None  # Для внешних ключей
    on_delete: Optional[str] = None  # Для внешних ключей: CASCADE, NO ACTION и т.д.
    options: Optional[str] = None
    enum_values: Optional[list[str]] = None  # Для перечисляемых и флагов
    pg_type: Optional[str] = None  # PostgreSQL-специфичный тип (jsonb и т.д.)

    # Поля, обогащённые из БД
    db_column_name: Optional[str] = None  # Реальное имя колонки в БД
    db_data_type: Optional[str] = None  # Реальный тип данных в БД
    db_is_nullable: Optional[bool] = None
    db_column_default: Optional[str] = None
    db_comment: Optional[str] = None


@dataclass
class IndexInfo:
    """Описание индекса таблицы."""
    name: str
    fields: str  # Строка с перечислением полей
    expression: Optional[str] = None  # Условие WHERE
    index_type: Optional[str] = None  # gin, btree и т.д.
    options: Optional[str] = None  # UNIQUE, CLUSTERED и т.д.
    include_fields: Optional[str] = None  # Дополнительные поля (ДопПоля)


@dataclass
class TriggerInfo:
    """Описание триггера."""
    name: str
    version: Optional[int] = None
    template: Optional[str] = None
    file: Optional[str] = None


@dataclass
class PolicyInfo:
    """Описание политики RLS."""
    name: str
    file: Optional[str] = None


@dataclass
class ForeignKeyRelation:
    """Описание связи между таблицами через внешний ключ."""
    field_name: str
    source_table: str
    target_table: str
    on_delete: Optional[str] = None


@dataclass
class TableInfo:
    """Полное описание таблицы из xdic + данные из БД."""
    name: str
    description: Optional[str] = None
    category: Optional[str] = None  # Категория: Служебная, Журнал и т.д.
    view_type: Optional[str] = None  # Вид: Служебная, Временная и т.д.
    schema: Optional[str] = None  # Схема: tmp и т.д.

    fields: list[FieldInfo] = field(default_factory=list)
    indexes: list[IndexInfo] = field(default_factory=list)
    triggers: list[TriggerInfo] = field(default_factory=list)
    policies: list[PolicyInfo] = field(default_factory=list)
    foreign_keys: list[ForeignKeyRelation] = field(default_factory=list)

    # Данные из БД
    db_table_name: Optional[str] = None  # Реальное имя в БД
    db_schema: Optional[str] = None
    db_row_count: Optional[int] = None
    db_size_bytes: Optional[int] = None
    db_comment: Optional[str] = None

    @property
    def is_temporary(self) -> bool:
        return self.name.startswith("#") or self.view_type in ("Временная", "ВременнаяДляПользователя")

    @property
    def is_service(self) -> bool:
        return self.view_type == "Служебная" or self.category == "Служебная"


@dataclass
class ViewInfo:
    """Описание представления (VIEW)."""
    name: str
    file: Optional[str] = None
    version: Optional[int] = None
    fields: list[FieldInfo] = field(default_factory=list)


@dataclass
class FunctionInfo:
    """Описание функции или процедуры."""
    name: str
    file: Optional[str] = None
    version: Optional[int] = None
    is_procedure: bool = False
    dbms: Optional[str] = None  # MSSQL, PostgreSQL


@dataclass
class DatabaseSchema:
    """Корневой объект — полная схема базы данных."""
    version: Optional[str] = None
    tables: dict[str, TableInfo] = field(default_factory=dict)
    views: dict[str, ViewInfo] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)

    def get_table(self, name: str) -> Optional[TableInfo]:
        """Получить таблицу по имени (case-insensitive)."""
        name_lower = name.lower()
        for key, table in self.tables.items():
            if key.lower() == name_lower:
                return table
        return None

    def get_regular_tables(self) -> list[TableInfo]:
        """Получить только обычные (не временные, не служебные) таблицы."""
        return [t for t in self.tables.values() if not t.is_temporary]

    def get_tables_with_description(self) -> list[TableInfo]:
        """Получить таблицы, у которых есть описание."""
        return [t for t in self.tables.values() if t.description and not t.is_temporary]

    def find_tables_by_keyword(self, keyword: str) -> list[TableInfo]:
        """Поиск таблиц по ключевому слову в имени или описании."""
        keyword_lower = keyword.lower()
        results = []
        for table in self.tables.values():
            if keyword_lower in table.name.lower():
                results.append(table)
            elif table.description and keyword_lower in table.description.lower():
                results.append(table)
        return results

    def get_relations_for_table(self, table_name: str) -> list[ForeignKeyRelation]:
        """Получить все связи для указанной таблицы (входящие и исходящие)."""
        relations = []
        for table in self.tables.values():
            for fk in table.foreign_keys:
                if fk.source_table.lower() == table_name.lower() or \
                   fk.target_table.lower() == table_name.lower():
                    relations.append(fk)
        return relations
