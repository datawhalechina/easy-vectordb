// ========================================================================================
// 中式烹饪知识图谱构建脚本
// ========================================================================================

// 删除已存在的约束
DROP CONSTRAINT recipe_id_unique IF EXISTS;
DROP CONSTRAINT ingredient_id_unique IF EXISTS;
DROP CONSTRAINT cookingstep_id_unique IF EXISTS;
DROP CONSTRAINT cookingmethod_id_unique IF EXISTS;
DROP CONSTRAINT cookingtool_id_unique IF EXISTS;
DROP CONSTRAINT difficultylevel_id_unique IF EXISTS;

// 创建核心约束
CREATE CONSTRAINT recipe_id_unique IF NOT EXISTS FOR (r:Recipe) REQUIRE r.nodeId IS UNIQUE;
CREATE CONSTRAINT ingredient_id_unique IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.nodeId IS UNIQUE;
CREATE CONSTRAINT cookingstep_id_unique IF NOT EXISTS FOR (s:CookingStep) REQUIRE s.nodeId IS UNIQUE;

// 创建基础索引
CREATE INDEX recipe_name_index IF NOT EXISTS FOR (r:Recipe) ON (r.name);
CREATE INDEX recipe_difficulty_index IF NOT EXISTS FOR (r:Recipe) ON (r.difficulty);
CREATE INDEX recipe_cuisine_index IF NOT EXISTS FOR (r:Recipe) ON (r.cuisineType);
CREATE INDEX ingredient_name_index IF NOT EXISTS FOR (i:Ingredient) ON (i.name);
CREATE INDEX ingredient_category_index IF NOT EXISTS FOR (i:Ingredient) ON (i.category);
CREATE INDEX cookingstep_number_index IF NOT EXISTS FOR (s:CookingStep) ON (s.stepNumber);

// 创建全文搜索索引（包含更多字段）
CREATE FULLTEXT INDEX recipe_fulltext_index IF NOT EXISTS FOR (r:Recipe) ON EACH [r.name, r.description, r.preferredTerm, r.fsn, r.tags];
CREATE FULLTEXT INDEX ingredient_fulltext_index IF NOT EXISTS FOR (i:Ingredient) ON EACH [i.name, i.description, i.preferredTerm, i.fsn];
CREATE FULLTEXT INDEX cookingstep_fulltext_index IF NOT EXISTS FOR (s:CookingStep) ON EACH [s.name, s.description, s.methods, s.tools];

// 创建层次结构节点
RETURN 'Creating hierarchy nodes from nodes.csv';
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row
WHERE row.nodeId < '200000000'  // 层次结构节点的ID范围
  AND row.nodeId IS NOT NULL
  AND row.name IS NOT NULL
WITH row, row.labels as nodeType
CALL apoc.create.node([nodeType], {
    nodeId: row.nodeId,
    name: row.name,
    preferredTerm: row.preferredTerm,
    fsn: row.fsn,
    conceptType: row.conceptType,
    synonyms: row.synonyms,
    category: row.category,
    description: row.description,
    originalLabels: row.labels,
    isHierarchyNode: true
}) YIELD node
RETURN count(node);

// 创建菜谱节点
RETURN 'Creating Recipe nodes from nodes.csv';
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row
WHERE row.labels = 'Recipe'
  AND row.nodeId >= '200000000'  // 只处理具体实例，不处理层次结构
  AND row.nodeId IS NOT NULL
  AND row.name IS NOT NULL
    
    MERGE (r:Recipe {nodeId: row.nodeId})
    SET r.name = row.name,
        r.preferredTerm = row.preferredTerm,
        r.fsn = row.fsn,
        r.description = row.description,
        r.difficulty = CASE
            WHEN row.difficulty IS NOT NULL THEN toFloat(row.difficulty)
            ELSE null
        END,
        r.category = row.category,
        r.conceptType = row.conceptType,
        r.cuisineType = row.cuisineType,
        r.prepTime = row.prepTime,
        r.cookTime = row.cookTime,
        r.servings = row.servings,
        r.tags = row.tags,
        r.filePath = row.filePath,
        r.synonyms = row.synonyms,
        r.textForEmbedding = row.name + ' ' + COALESCE(row.preferredTerm, '') + ' ' +
                            COALESCE(row.description, '') + ' ' + COALESCE(row.tags, ''),
        r.originalLabels = row.labels;

// 创建食材节点
RETURN 'Creating Ingredient nodes';
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row
WHERE row.labels = 'Ingredient'
  AND row.nodeId >= '200000000'  // 只处理具体实例
  AND row.nodeId IS NOT NULL
  AND row.name IS NOT NULL
    
    MERGE (i:Ingredient {nodeId: row.nodeId})
    SET i.name = row.name,
        i.preferredTerm = row.preferredTerm,
        i.fsn = row.fsn,
        i.description = row.description,
        i.category = row.category,
        i.conceptType = row.conceptType,
        i.amount = row.amount,
        i.unit = row.unit,
        i.isMain = CASE
            WHEN row.isMain IS NOT NULL THEN toBoolean(row.isMain)
            ELSE null
        END,
        i.synonyms = row.synonyms,
        i.originalLabels = row.labels;

// 创建烹饪步骤节点
RETURN 'Creating CookingStep nodes';
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row
WHERE row.labels = 'CookingStep'
  AND row.nodeId >= '200000000'  // 只处理具体实例
  AND row.nodeId IS NOT NULL
  AND row.name IS NOT NULL
    
    MERGE (s:CookingStep {nodeId: row.nodeId})
    SET s.name = row.name,
        s.preferredTerm = row.preferredTerm,
        s.fsn = row.fsn,
        s.description = row.description,
        s.category = row.category,
        s.conceptType = row.conceptType,
        s.stepNumber = CASE
            WHEN row.stepNumber IS NOT NULL THEN toFloat(row.stepNumber)
            ELSE null
        END,
        s.methods = row.methods,
        s.tools = row.tools,
        s.timeEstimate = row.timeEstimate,
        s.synonyms = row.synonyms,
        s.originalLabels = row.labels;

// 创建其他类型节点
RETURN 'Creating other node types';
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row
WHERE NOT (row.labels IN ['Recipe', 'Ingredient', 'CookingStep'])
  AND row.nodeId IS NOT NULL
  AND row.name IS NOT NULL
WITH row, row.labels as nodeType
CALL apoc.create.node([nodeType], {
    nodeId: row.nodeId,
    name: row.name,
    description: row.description,
    category: row.category,
    conceptType: row.conceptType,
    originalLabels: row.labels
}) YIELD node
RETURN count(node);

// 创建需要关系 (801000001)
RETURN 'Creating REQUIRES relationships';
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row
WHERE row.relationshipType = '801000001'
  AND row.startNodeId IS NOT NULL
  AND row.endNodeId IS NOT NULL
MATCH (source:Recipe {nodeId: row.startNodeId})
MATCH (target:Ingredient {nodeId: row.endNodeId})
MERGE (source)-[r:REQUIRES]->(target)
SET r.relationshipId = row.relationshipId,
    r.amount = row.amount,
    r.unit = row.unit,
    r.originalType = row.relationshipType;

// 创建包含步骤关系 (801000003)
RETURN 'Creating CONTAINS relationships';
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row
WHERE row.relationshipType = '801000003'
  AND row.startNodeId IS NOT NULL
  AND row.endNodeId IS NOT NULL
MATCH (source:Recipe {nodeId: row.startNodeId})
MATCH (target:CookingStep {nodeId: row.endNodeId})
MERGE (source)-[r:CONTAINS_STEP]->(target)
SET r.relationshipId = row.relationshipId,
    r.stepOrder = CASE
        WHEN row.step_order IS NOT NULL AND row.step_order <> ''
        THEN toFloat(row.step_order)
        ELSE null
    END,
    r.originalType = row.relationshipType;

// 创建其他关系类型
RETURN 'Creating other relationships';
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row
WHERE NOT (row.relationshipType IN ['801000001', '801000003'])
  AND row.relationshipType IS NOT NULL
  AND row.startNodeId IS NOT NULL
  AND row.endNodeId IS NOT NULL
MATCH (source {nodeId: row.startNodeId})
MATCH (target {nodeId: row.endNodeId})
WITH source, target, row,
     CASE row.relationshipType
         WHEN '801000004' THEN 'BELONGS_TO'
         WHEN '801000005' THEN 'DIFFICULTY_LEVEL'
         ELSE row.relationshipType
     END as relType
CALL apoc.create.relationship(source, relType, {
    relationshipId: row.relationshipId,
    amount: row.amount,
    unit: row.unit,
    stepOrder: CASE
        WHEN row.step_order IS NOT NULL AND row.step_order <> ''
        THEN toFloat(row.step_order)
        ELSE null
    END,
    originalType: row.relationshipType
}, target) YIELD rel
RETURN count(rel);

// 基于数据创建分类节点（支持多重分类）
RETURN 'Creating category nodes from data';

// 处理多重分类：按逗号分割并创建分类节点
MATCH (n)
WHERE n.category IS NOT NULL AND n.category <> ''
WITH n, split(n.category, ',') as categoryList
UNWIND categoryList as categoryName
WITH n, trim(categoryName) as cleanCategoryName
WHERE cleanCategoryName <> ''
MERGE (cat:Category {name: cleanCategoryName})
MERGE (n)-[:BELONGS_TO_CATEGORY]->(cat);

// 为每个唯一的conceptType创建概念类型节点
MATCH (n)
WHERE n.conceptType IS NOT NULL AND n.conceptType <> ''
WITH DISTINCT n.conceptType as conceptTypeName
MERGE (ct:ConceptType {name: conceptTypeName});

// 连接实体到概念类型
MATCH (n), (ct:ConceptType)
WHERE n.conceptType = ct.name
MERGE (n)-[:HAS_CONCEPT_TYPE]->(ct);

// 创建基于数据的相似性关系（仅基于相同分类）
RETURN 'Creating data-driven similarity relationships';
MATCH (n1), (n2)
WHERE n1.category = n2.category 
  AND n1.nodeId <> n2.nodeId
  AND n1.category IS NOT NULL
  AND labels(n1) = labels(n2)
WITH n1, n2, n1.category as sharedCategory
LIMIT 1000
MERGE (n1)-[:SIMILAR {
    basis: 'same_category',
    category: sharedCategory,
    confidence: 0.6
}]->(n2);

// 生成统计信息
RETURN 'Generating statistics';
MATCH (n) 
WITH labels(n) as nodeTypes, count(n) as nodeCount
UNWIND nodeTypes as nodeType
WITH nodeType, sum(nodeCount) as totalCount
WHERE nodeType <> 'Category' AND nodeType <> 'ConceptType'
CREATE (stat:GraphStatistics {
    nodeType: nodeType,
    count: totalCount,
    generatedAt: datetime()
});

MATCH ()-[r]->()
WITH type(r) as relType, count(r) as relCount
CREATE (stat:RelationshipStatistics {
    relationshipType: relType,
    count: relCount,
    generatedAt: datetime()
});

// 处理同义词关系（如果数据中有同义词）
RETURN 'Creating synonym relationships';
MATCH (n)
WHERE n.synonyms IS NOT NULL AND n.synonyms <> '' AND n.synonyms <> '[]'
WITH n, n.synonyms as synonymsJson
WHERE synonymsJson IS NOT NULL
// 这里可以根据实际的同义词格式进行解析
// 目前先创建一个标记，表示该节点有同义词
SET n.hasSynonyms = true;

// 创建基于时间的步骤序列关系
RETURN 'Creating step sequence relationships';
MATCH (r:Recipe)-[:CONTAINS_STEP]->(s1:CookingStep)
MATCH (r)-[:CONTAINS_STEP]->(s2:CookingStep)
WHERE s1.stepNumber IS NOT NULL AND s2.stepNumber IS NOT NULL
  AND s2.stepNumber = s1.stepNumber + 1
MERGE (s1)-[:NEXT_STEP {
    sequence: true,
    timeDifference: s2.stepNumber - s1.stepNumber
}]->(s2);

// 创建基于工具的关系
RETURN 'Creating tool-based relationships';
MATCH (s1:CookingStep), (s2:CookingStep)
WHERE s1.tools IS NOT NULL AND s2.tools IS NOT NULL
  AND s1.tools = s2.tools
  AND s1.nodeId <> s2.nodeId
WITH s1, s2, s1.tools as sharedTool
LIMIT 1000
MERGE (s1)-[:USES_SAME_TOOL {
    tool: sharedTool,
    similarity: 0.5
}]->(s2);

// 创建基于烹饪方法的关系
RETURN 'Creating method-based relationships';
MATCH (s1:CookingStep), (s2:CookingStep)
WHERE s1.methods IS NOT NULL AND s2.methods IS NOT NULL
  AND s1.methods = s2.methods
  AND s1.nodeId <> s2.nodeId
WITH s1, s2, s1.methods as sharedMethod
LIMIT 1000
MERGE (s1)-[:USES_SAME_METHOD {
    method: sharedMethod,
    similarity: 0.7
}]->(s2);

// 为难度等级创建特殊处理
RETURN 'Processing difficulty levels';
MATCH (d:DifficultyLevel)
SET d.level = CASE d.name
    WHEN '一星' THEN 1
    WHEN '二星' THEN 2
    WHEN '三星' THEN 3
    WHEN '四星' THEN 4
    WHEN '五星' THEN 5
    ELSE 0
END;

// 创建菜谱到难度等级的直接关系（如果数据中有对应关系）
MATCH (r:Recipe)
WHERE r.difficulty IS NOT NULL
MATCH (d:DifficultyLevel)
WHERE d.level = toInteger(r.difficulty)
MERGE (r)-[:HAS_DIFFICULTY_LEVEL]->(d);

// 优化：为高频查询创建计算属性
RETURN 'Creating computed properties for optimization';

// 为菜谱计算总时间
MATCH (r:Recipe)
WHERE r.prepTime IS NOT NULL AND r.cookTime IS NOT NULL
SET r.totalTime = CASE
    WHEN r.prepTime =~ '.*分钟.*' AND r.cookTime =~ '.*分钟.*' THEN
        toInteger(split(r.prepTime, '分钟')[0]) + toInteger(split(r.cookTime, '分钟')[0])
    ELSE null
END;

// 为菜谱计算食材数量
MATCH (r:Recipe)
OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
WITH r, count(i) as ingredientCount
SET r.ingredientCount = ingredientCount;

// 为菜谱计算步骤数量
MATCH (r:Recipe)
OPTIONAL MATCH (r)-[:CONTAINS_STEP]->(s:CookingStep)
WITH r, count(s) as stepCount
SET r.stepCount = stepCount;

RETURN 'Knowledge graph construction completed!';

MATCH (n)
RETURN labels(n) as NodeType, count(n) as Count
ORDER BY Count DESC;