# 基于智能HTML预处理和内容容器定位的XPath生成工具

适用于爬虫开发，自动生成XPath表达式
非常方便自动化极高的爬虫工具！！！！！！！！！！！！！！

本项目采用先进的HTML预处理和智能内容容器定位策略，从网页中定位包含事项列表的最优父级容器，并生成对应的XPath表达式。


下面时开发笔记：
---
一个页面中，存在k个列表，假定k=3，有三个列表，列表1为导航栏，里面有8个列表项，列表2为侧边栏，里面有5个列表项，列表3是事项列表，里面有7个列表项，
此时，我的代码会把列表1作为目标获取，但实际情况应该是列表3才是正确的，这怎么办呢，
目前对于目标列表3，可能存在以下特点：里面往往存在时间字符串，并且有些页面中的文字的长度是大于列表1和列表2的。
除此之外，对于列表1，也有以下特点：当我们误获取这个列表1的时候，会去处理组装他的xpath，这个xpath里面往往是存在nav三个字母的，
在我的观察下，大部分情况中，只要xpath里面包含nav，那就很大可能说明获取失败了，没有获取到列表3，而是获取到了列表1

对于js页面，name中名称一定要准确，并且！name要尽量要少一点，比如“法定主动公开内容”，这个就写“法定”即可，这俩字有代表性，不能写“内容”这俩字，没有任何的代表性


v3.0
首先，保持v2版本的处理逻辑，赋分和分层过滤等思路不变，v2版本的目前可以正确的识别列表，并且排除干扰列表，所以，对于v2版本，问题是赋分的逻辑存在问题，导致没有列表的url中识别到header和footer中，
对于v3版本，将在v2版本的基础上增加html过滤操作，过滤掉hearder和footer的html容器，然后得到一个干净的html容器传入到v2分层过滤中，这样可以排除掉不必要的干扰，不会影响赋分的逻辑。

对于如何过滤header和footer，有以下的思路:首先header和footer中的一定包含了某些固定的文字内容，比如header里面肯定有首页或者无障碍浏览等文字，我们获取到这个文字后，回溯倒着去找父标签，一直去找，他的父容器的前一级或者两级祖先容器大概率class里面包含head文字字段
然后继续找父级容器，直到，父级容器里面出现了meta或者html等表示整个html的标签，说明已经到顶了，此时，就可以删除掉这个容器了，这样就过滤掉了header容器，同理，过滤掉footer容器，最后得到一份干净的不包含head和foot干扰的HTML内容。
对于这个干净的html，我们就可以进入v2版本的处理流程中，然后对里面的内容进行赋分判断，密度识别。 

最后，写个下一个使用者：
最终过滤的代码也是需要修改的，不一定要获取最精确的容器，也不要获取最宽泛的容器，我的场景大部分都是政务网站，所以项目的赋分机制，过滤机制，以及最终的容器选择机制，还有一些xpath以及文字的识别都是有一定的针对性的，不过，我觉得大部分的网站都是这个样子吧，所以，这个项目应该可以满足大部分的使用场景。
---

#xpathFake.py #xpathDs.py xpathFake代码的效果比xpathDs的效果要好
1. 更简洁的算法流程
xpathFake.py：

直接使用 find_article_container() → find_main_content_in_cleaned_html()
流程简单清晰，减少了过度处理的风险
xpathDs.py：

增加了 perform_second_level_cleaning() 二次清理步骤
流程更复杂，可能过度清理导致丢失有效内容
2. 容器选择策略的差异
xpathFake.py 在 find_main_content_in_cleaned_html() 中：

# 选择得分最高的容器
scored_containers.sort(key=lambda x: x[1], reverse=True)
best_score = scored_containers[0][1]
same_score_containers = [container for container, score in scored_containers if score == best_score]
if len(same_score_containers) > 1:
    # 选择层级最深的一个（儿子容器）
    best_container = select_best_from_same_score_containers(same_score_containers)
else:
    best_container = scored_containers[0][0]
xpathDs.py 中：

# 直接选择得分最高的，注释掉了层级选择逻辑
best_container = scored_containers[0][0]
3. HTML输出调试的差异
xpathFake.py：

# 输出清理后的HTML到终端
cleaned_html = html.tostring(body, encoding='unicode', pretty_print=True)
print("\n=== 清理后的HTML内容 ===")
print(cleaned_html[:2000] + "..." if len(cleaned_html) > 2000 else cleaned_html)
xpathDs.py：

# 注释掉了HTML输出
# cleaned_html = html.tostring(body, encoding='unicode', pretty_print=True)
# print("\n=== 清理后的HTML内容 ===")
4. 二次清理的影响
xpathDs.py
 中的 perform_second_level_cleaning() 可能过于激进：

最多移除2层DOM结构
可能误删包含目标内容的容器
增加了算法复杂度和出错概率
5. 容器深度选择逻辑
xpathFake.py
 保留了 select_best_from_same_score_containers() 函数，能够：

在得分相同的容器中选择层级最深的
更精确地定位到实际内容容器
避免选择过于宽泛的父容器


## 核心算法：智能HTML预处理 + 内容容器定位

### 算法流程

1. **HTML预处理与干扰项排除**
   ```python
   def preprocess_html_remove_interference(page_tree):
       # 第一步：通过内容特征回溯删除首部和尾部容器
       body = remove_header_footer_by_content_traceback(body)
       
       # 第二步：识别并删除明显的页面级header/footer容器
       interference_containers = []
       for container in all_containers:
           if is_interference_container(container):
               interference_containers.append(container)
   ```

2. **内容特征回溯过滤**
   ```python
   def remove_header_footer_by_content_traceback(body):
       # 首部内容特征关键词：登录、注册、首页、导航等
       # 尾部内容特征关键词：版权所有、备案号、技术支持等
       # 通过内容特征关键词查找元素并回溯父容器进行删除
   ```

3. **智能容器评分机制**
   ```python
   def calculate_content_container_score(container):
       # 1. 基础内容长度评分（长内容+50，中等内容+35，短内容+20）
       # 2. 时间特征检测（强正面特征，+25/匹配）
       # 3. 正面类名/ID特征（content、main、article等，+20/匹配）
       # 4. 结构化内容检测（p、h1、h2、li等，+2/元素）
       # 5. 图片内容（+3/图片）
       # 6. 负面特征检测（sidebar、aside、ad等，-30）
   ```

4. **层级深度优化选择**
   ```python
   def select_best_from_same_score_containers(containers):
       # 从得分相同的多个容器中选择层级最深的一个（儿子容器）
       # 计算容器距离body的层级深度，选择最深的容器
   ```

## 项目结构

```
GetXpath/
├── xpathFake.py          # 核心处理模块（主文件）
├── webdriver_pool.py     # WebDriver池管理
├── test.yml              # 输入测试文件
├── testout.yml           # 结果输出文件
├── waitprocess/          # 待处理文件目录
├── processed/           # 已处理文件目录
└── README.md            # 项目文档
```

## 输入输出格式

### 输入格式 (test.yml)
```yaml
name: 示例网站
url: https://example.com/list-page
---
name: JS页面示例（需要点击）
url: https://example.com/js-page
```

### 输出格式 (testout.yml)
```yaml
---
name: 示例网站
url: https://example.com/list-page
xpath: "//div[@class='content']"
xpathList4Click: []
---
name: JS页面示例
url: https://example.com/js-page  
xpath: "//section[@id='main-content']"
xpathList4Click:
  - "//button[contains(text(),'加载更多')]"
  - "//div[@class='tab' and contains(text(),'列表')]"
```

## 核心功能

### 双引擎网页获取
- **Selenium引擎**：支持JavaScript渲染页面，自动等待和重试
- **Requests后备引擎**：Selenium失败时自动切换
- **DrissionPage支持**：专门处理需要交互的JS页面

```python
def get_html_content_Selenium(url, max_retries=3):
    # Selenium获取，失败时回退到requests
def get_html_content_Drission(name, url):
    # 处理需要点击操作的JS页面
```

### 智能HTML预处理
- **内容特征回溯过滤**：通过关键词识别并删除header/footer
- **结构特征识别**：识别nav、menu、sidebar等干扰容器
- **多层容器处理**：支持body标签内多层div嵌套的情况

### 精准内容容器定位
- **多维度评分系统**：内容长度、时间特征、结构化内容等
- **层级深度优化**：优先选择最深层的儿子容器
- **相同得分处理**：通过层级深度解决多个容器得分相同的问题

### 智能XPath生成
- **优先级策略**：ID > 类名 > 属性 > 位置
- **验证机制**：确保生成的XPath有效且包含足够内容

## 算法优势

- **精准干扰排除**：通过内容和结构双重特征识别header/footer
- **抗干扰能力**：有效过滤导航栏、菜单、侧边栏等干扰容器  
- **JS页面支持**：专门处理需要点击操作的JavaScript渲染页面
- **自适应优化**：层级深度选择确保定位到最合适的容器
- **健壮性**：支持多种页面结构，自动重试和后备机制

## 使用说明

### 基本使用
```bash
# 处理单个test.yml文件
python xpathFake.py
```

### 批量处理
```python
# 处理waitprocess目录下的所有yml文件
input_folder = "waitprocess"
files = glob.glob(os.path.join(input_folder, "*.yml"))
for input_file in files:
    process_yml_file(input_file, output_file)
```

### JS页面处理
对于需要点击操作的JS页面，在name中标注"js"后缀，系统会自动使用DrissionPage处理：
- name要尽量简短且有代表性（如"法定"而不是"内容"）
- 系统会自动识别并点击相关标签加载内容

## 版本历史

### v2.0 (2025.8.22)
- 修改算法逻辑，从提取列表改为提取正文所在容器
- 增强HTML预处理，排除头部导航和底部footer
- 改进容器评分机制，专注于识别真正的内容区域
- 增加层级深度优化选择，解决多个容器得分相同的问题

### v1.0
- 初始版本，专注于列表项提取
- 基于时间特征和文本长度识别目标列表

## 性能优化

### 智能重试机制
```python
for attempt in range(max_retries):
    try:
        # 尝试获取
    except:
        if attempt == max_retries - 1:
            return get_html_content(url)  # 后备方案
```

### 并行处理支持
```python
# 支持多线程并行处理
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(process_entry, entries))
```

### 验证机制
```python
def validate_xpath(xpath, html_content):
    results = tree.xpath(xpath)
    # 验证是否找到足够的内容
    if len(list_items) >= 3: 
        return True
```
        