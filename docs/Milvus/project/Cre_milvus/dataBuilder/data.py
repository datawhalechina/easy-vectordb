from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .tools.csvmake import process_csv
from .tools.mdmake import process_md
from .tools.pdfmake import process_pdf
from .tools.txtmake import process_txt
from System.monitor import log_event
import traceback

def data_process(data_location, url_split):
    """
    data_location: 用户上传的文件夹路径
    url_split: 是否对文本做url切分
    自动识别文件夹下的csv、md、pdf、txt文件并多线程处理
    """
    dataList = []
    folder = Path(data_location)
    
    files = [f for f in folder.rglob("*") if f.is_file()]
    print(f"找到 {len(files)} 个文件")
    

    valid_extensions = {".csv", ".md", ".pdf", ".txt"}
    tasks = []
    
    for file in files:
        ext = file.suffix.lower()
        if ext in valid_extensions:
            tasks.append((ext[1:], str(file)))  
    
    print(f"准备处理 {len(tasks)} 个有效文件")
    
    def process_one(task):
        file_type, file_path = task
        print(f"开始处理文件: {file_path}, 类型: {file_type}")
        try:
            if file_type == "csv":
                return process_csv(csv_path=file_path)
            elif file_type == "md":
                return process_md(md_file_path=file_path, url_split=url_split)
            elif file_type == "pdf":
                return process_pdf(pdf_path=file_path, url_split=url_split)
            elif file_type == "txt":
                return process_txt(txt_path=file_path, url_split=url_split)
        except Exception as e:
            print(f"处理文件出错: {file_path}\n错误详情: {traceback.format_exc()}")
            return []  # 返回空列表而不是None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_one, tasks):
            # 确保所有处理函数都返回列表
            if isinstance(result, list):
                dataList.extend(result)
            elif result is not None:
                # 如果返回单个对象，包装成列表
                dataList.append(result)
    
    print(f"处理完成，共处理了 {len(dataList)} 条数据。")
    log_event(f"数据处理完成，共处理 {len(tasks)} 个文件，生成 {len(dataList)} 条数据")
    return dataList