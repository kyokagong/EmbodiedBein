# 数据集本体关联展示与训练筛选

- 数据列表和详情显示 `embodiment_name`、`embodiment_id` 与 `embodiment_env_id`。
- 新建训练页只列出所选本体环境下的数据集；切换模板或本体会清空旧选择。
- `_build_spec` 强制数据集的 `embodiment_env_id` 与模板环境一致，页面校验不可绕过。
- 复用 `datasets.embodiment_env_id`，没有新增表或数据库迁移。
