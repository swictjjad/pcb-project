"""
软著申请材料 Markdown 转 Word 脚本
"""
import os
import re
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn

MATERIAL_DIR = r"c:\Users\Uaena\Desktop\PCB_Defect_Detection\PCB_缺陷检测系统_项目\PCB_缺陷检测系统\pcb_defect_system\results\软著申请材料_20260609"
OUTPUT_DIR = MATERIAL_DIR

def set_chinese_font(run, font_name='宋体', font_size=12):
    """设置中文字体"""
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)

def add_heading_with_style(doc, text, level):
    """添加标题并设置样式"""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        if level == 1:
            run.font.size = Pt(18)
            run.font.bold = True
        elif level == 2:
            run.font.size = Pt(16)
            run.font.bold = True
        elif level == 3:
            run.font.size = Pt(14)
            run.font.bold = True
        else:
            run.font.size = Pt(12)
    return heading

def add_paragraph_with_style(doc, text, font_size=12, bold=False, indent=False):
    """添加段落并设置样式"""
    para = doc.add_paragraph()
    if indent:
        para.paragraph_format.first_line_indent = Cm(0.74)  # 两字符缩进
    para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    
    run = para.add_run(text)
    set_chinese_font(run, '宋体', font_size)
    run.font.bold = bold
    return para

def parse_table(lines, start_idx):
    """解析 Markdown 表格"""
    rows = []
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if not line or not line.startswith('|'):
            break
        # 跳过分隔行
        if re.match(r'^\|[\s\-:]+\|', line):
            i += 1
            continue
        # 解析单元格
        cells = [c.strip() for c in line.split('|')[1:-1]]
        rows.append(cells)
        i += 1
    return rows, i

def add_table_from_rows(doc, rows):
    """从行数据创建 Word 表格"""
    if not rows:
        return None
    
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    for i, row_data in enumerate(rows):
        row = table.rows[i]
        for j, cell_text in enumerate(row_data):
            cell = row.cells[j]
            cell.text = cell_text
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    set_chinese_font(run, '宋体', 10)
                    if i == 0:  # 表头加粗
                        run.font.bold = True
    return table

def convert_md_to_docx(md_path, docx_path, title):
    """将 Markdown 文件转换为 Word 文档"""
    print(f"正在转换: {os.path.basename(md_path)}")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    doc = Document()
    
    # 设置页面边距
    sections = doc.sections
    for section in sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.17)
        section.right_margin = Cm(3.17)
    
    # 添加标题
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(title)
    title_run.font.name = '黑体'
    title_run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    
    doc.add_paragraph()  # 空行
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 跳过空行
        if not stripped:
            i += 1
            continue
        
        # 跳过水平线
        if stripped.startswith('---'):
            i += 1
            continue
        
        # 标题
        if stripped.startswith('#'):
            match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if match:
                level = len(match.group(1))
                text = match.group(2)
                add_heading_with_style(doc, text, min(level, 3))
                i += 1
                continue
        
        # 表格
        if stripped.startswith('|'):
            rows, new_i = parse_table(lines, i)
            if rows:
                add_table_from_rows(doc, rows)
                doc.add_paragraph()  # 表格后空行
            i = new_i
            continue
        
        # 代码块
        if stripped.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过结束的 ```
            
            if code_lines:
                code_para = doc.add_paragraph()
                code_para.paragraph_format.left_indent = Cm(0.5)
                for code_line in code_lines:
                    run = code_para.add_run(code_line + '\n')
                    run.font.name = 'Consolas'
                    run.font.size = Pt(9)
            continue
        
        # 普通段落
        # 处理加粗和行内代码
        text = stripped
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # 暂时简化处理
        text = re.sub(r'`(.+?)`', r'\1', text)
        
        # 列表项
        if stripped.startswith('- ') or stripped.startswith('* '):
            text = '• ' + stripped[2:]
        
        add_paragraph_with_style(doc, text, font_size=12, indent=True)
        i += 1
    
    # 保存
    doc.save(docx_path)
    print(f"  已保存: {os.path.basename(docx_path)}")

def convert_source_code_to_docx(py_path, docx_path, title):
    """将源代码文件转换为 Word 文档"""
    print(f"正在转换源代码: {os.path.basename(py_path)}")
    
    with open(py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    doc = Document()
    
    # 设置页面边距
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.17)
        section.right_margin = Cm(3.17)
    
    # 标题
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(title)
    title_run.font.name = '黑体'
    title_run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    title_run.font.size = Pt(16)
    title_run.font.bold = True
    
    doc.add_paragraph()
    
    # 添加页眉信息
    header_para = doc.add_paragraph()
    header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header_run = header_para.add_run(f"PCB缺陷检测与智能分拣系统 V1.0  |  源代码  |  共 {len(lines)} 行")
    set_chinese_font(header_run, '宋体', 10)
    
    doc.add_paragraph()
    
    # 添加代码（每页约50行，分批添加）
    code_para = doc.add_paragraph()
    code_para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    
    for idx, line in enumerate(lines):
        # 添加行号
        line_with_num = f"{idx+1:5d}  {line.rstrip()}"
        run = code_para.add_run(line_with_num + '\n')
        run.font.name = 'Consolas'
        run.font.size = Pt(9)
    
    doc.save(docx_path)
    print(f"  已保存: {os.path.basename(docx_path)}")

def main():
    print("=" * 60)
    print("软著申请材料 Markdown → Word 转换工具")
    print("=" * 60)
    print()
    
    # 转换软件说明书
    md_file = os.path.join(MATERIAL_DIR, "01_软件说明书.md")
    docx_file = os.path.join(OUTPUT_DIR, "01_软件说明书.docx")
    if os.path.exists(md_file):
        convert_md_to_docx(md_file, docx_file, "PCB缺陷检测与智能分拣系统\n软件说明书（设计说明书）")
    
    # 转换申请表填写指南
    md_file = os.path.join(MATERIAL_DIR, "03_申请表填写指南.md")
    docx_file = os.path.join(OUTPUT_DIR, "03_申请表填写指南.docx")
    if os.path.exists(md_file):
        convert_md_to_docx(md_file, docx_file, "软件著作权登记申请材料清单与填写指南")
    
    # 转换训练结果汇总
    md_file = os.path.join(MATERIAL_DIR, "04_训练结果汇总.md")
    docx_file = os.path.join(OUTPUT_DIR, "04_训练结果汇总.docx")
    if os.path.exists(md_file):
        convert_md_to_docx(md_file, docx_file, "PCB缺陷检测模型训练结果汇总")
    
    # 转换源代码
    src_dir = os.path.join(MATERIAL_DIR, "02_源代码")
    
    py_file = os.path.join(src_dir, "软著源代码_第一部分.py")
    docx_file = os.path.join(OUTPUT_DIR, "02_源代码_第一部分.docx")
    if os.path.exists(py_file):
        convert_source_code_to_docx(py_file, docx_file, "软著源代码（第一部分）")
    
    py_file = os.path.join(src_dir, "软著源代码_第二部分.py")
    docx_file = os.path.join(OUTPUT_DIR, "02_源代码_第二部分.docx")
    if os.path.exists(py_file):
        convert_source_code_to_docx(py_file, docx_file, "软著源代码（第二部分）")
    
    py_file = os.path.join(src_dir, "软著源代码_第三部分.py")
    docx_file = os.path.join(OUTPUT_DIR, "02_源代码_第三部分.docx")
    if os.path.exists(py_file):
        convert_source_code_to_docx(py_file, docx_file, "软著源代码（第三部分）")
    
    print()
    print("=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    print("生成的 Word 文件：")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.docx'):
            print(f"  - {f}")

if __name__ == "__main__":
    main()
