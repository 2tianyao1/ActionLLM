from openpyxl import Workbook

def create_empty_excel_file(name):
    # 创建工作簿
    workbook = Workbook()

    # 获取默认的工作表
    sheet = workbook.active

    # 设置工作表的名称
    sheet.title = 'Sheet1'


    # 保存文件
    output_path = f"/home/tianyao/code/ty/LLMAction_after/LLMAction_01_4/result/per_sp/50s/resnet50_text_epo25_adapter4/excel/{name}.xlsx"  # 设置输出文件路径和文件名
    workbook.save(output_path)

    print(f"Excel file '{output_path}' created successfully.")

# 示例调用

# create_empty_excel_file("test")
create_empty_excel_file("bf_sp1")
create_empty_excel_file("bf_sp2")
create_empty_excel_file("bf_sp3")
create_empty_excel_file("bf_sp4")
create_empty_excel_file("bf_sp5")