import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'EasyVectorDB 教程',
  description: '向量数据库学习与实战指南',
  lang: 'zh-CN',
  base: '/easy-vectordb/',
  lastUpdated: true,
  ignoreDeadLinks: true,

  head: [
    // ['link', { rel: 'icon', href: '/images/image.png' }],
    ['meta', { name: 'theme-color', content: '#3c8772' }]
  ],

  markdown: {
    lineNumbers: true,
    mermaid: true,
    math: true,
    image: {
      lazyLoading: false
    }
  },

  vite: {
    assetsInclude: ['**/*.PNG', '**/*.JPG', '**/*.JPEG', '**/*.GIF', '**/*.SVG']
  },

  themeConfig: {
    // logo: '/images/image.png',

    returnToTopLabel: '返回顶部',

    // 控制右侧目录显示的标题层级
    outline: {
      level: [1, 4] // 显示从 # (h1) 到 #### (h4) 的标题
    },

    nav: [
      { text: '首页', link: '/' },
      { text: '向量基础', link: '/base/chapter1/Milvus向量数据库入门' },
      { text: 'Faiss 教程', link: '/Faiss/chapter1/FAISS入门与环境搭建' },
      { text: 'Milvus 教程', link: '/Milvus/chapter1/Milvus向量数据库入门' },
      { text: 'Annoy 教程', link: '/Annoy/chapter1/Annoy 介绍' },
      { text: '补充内容', link: '/more/chapter5/向量' },
      { text: '实战项目', link: '/projects/' }
    ],

    sidebar: {
      '/base/': [
        {
          text: 'Chapter 1 · 项目介绍',
          items: [
            { text: '项目介绍', link: '/base/chapter1/项目介绍' }
          ]
        },
        {
          text: 'Chapter 2 · 向量数据库概念',
          items: [
            { text: '为什么需要向量数据库', link: '/base/chapter2/为什么需要向量数据库' },
          ]
        },
        {
          text: 'Chapter 3 · 向量嵌入算法基础',
          items: [
            { text: '向量嵌入算法基础', link: '/base/chapter3/向量嵌入算法基础' },
          ]
        },
        {
          text: 'Chapter 4 · 向量搜索算法基础',
          items: [
            { text: '向量搜索算法基础', link: '/base/chapter4/向量搜索算法基础' },
          ]
        },
        {
          text: 'Chapter 5 · 向量ANN搜索算法',
          items: [
            { text: 'ANN搜索算法', link: '/base/chapter5/ANN搜索算法' },
            { text: 'IVF算法', link: '/base/chapter5/IVF算法' },
            { text: 'HNSW算法', link: '/base/chapter5/HNSW算法' },
            { text: 'PQ算法', link: '/base/chapter5/PQ算法' },
            { text: 'LSH算法', link: '/base/chapter5/LSH算法' },
          ]
        },
        {
          text: 'Chapter 6 · 实现你自己的向量数据库',
          items: [
            { text: '实现你自己的向量数据库', link: '/base/chapter6/实现你自己的向量数据库' },
          ]
        },
      ],

      '/Faiss/': [
        {
          text: 'Chapter 1 · FAISS入门与环境搭建',
          items: [
            { text: 'FAISS 入门与环境搭建', link: '/Faiss/chapter1/FAISS入门与环境搭建' }
          ]
        },
        {
          text: 'Chapter 2 · FAISS数据结构与索引类型',
          items: [
            { text: 'FAISS数据结构与索引类型', link: '/Faiss/chapter2/FAISS数据结构与索引' }
          ]
        },
        {
          text: 'Chapter 3 · FAISS核心功能进阶',
          items: [
            { text: 'FAISS核心功能进阶', link: '/Faiss/chapter3/FAISS核心功能进阶' }
          ]
        },
        {
          text: 'Chapter 4 · FAISS性能调优与评估',
          items: [
            { text: 'FAISS性能调优与评估', link: '/Faiss/chapter4/FAISS性能调优与评估' }
          ]
        },
        {
          text: 'Chapter 5 · FAISS工程化落地实战',
          items: [
            { text: 'FAISS工程化落地实战', link: '/Faiss/chapter5/FAISS工程化落地实战' }
          ]
        },
      ],
      '/Milvus/': [
        {
          text: 'Chapter 1 · Milvus向量数据库入门',
          items: [
            { text: 'Milvus向量数据库入门', link: '/Milvus/chapter1/Milvus向量数据库入门' }
          ]
        },
        {
          text: 'Chapter 2 · Milvus核心概念',
          items: [
            { text: 'Milvus核心概念', link: '/Milvus/chapter2/Milvus核心概念' }
          ]
        },
        {
          text: 'Chapter 3 · Milvus基础操作',
          items: [
            { text: 'Milvus基础操作', link: '/Milvus/chapter3/PyMilvus核心API实战' }
          ]
        },
        {
          text: 'Chapter 4 · 综合实战',
          items: [
            { text: '综合实战', link: '/Milvus/chapter4/综合实战.md' }
          ]
        }
        ,
        {
          text: 'Chapter 5 · Milvus生态与进阶学习',
          items: [
            { text: 'Milvus 生态与进阶', link: '/Milvus/chapter5/Milvus生态与进阶学习' },
            { text: 'Milvus Lite部署教程', link: '/Milvus/chapter5/Milvus生态与进阶学习' },
            { text: 'Milvus Standalone部署教程', link: '/Milvus/chapter5/Milvus生态与进阶学习' },
            { text: 'MinerU 部署教程', link: '/Milvus/chapter5/MinerU部署教程' }
          ]
        }
      ],
      '/Annoy/': [
        {
          text: 'Chapter 1 · 基础概念',
          items: [
            { text: 'Annoy 介绍', link: '/Annoy/chapter1/Annoy 介绍' },
          ]
        },
        {
          text: 'Chapter 2 · 部署与基础使用',
          items: [
            { text: 'Annoy 部署与基础使用', link: '/Annoy/chapter2/Annoy 部署与基础使用' },
          ]
        },
      ],
      '/more/': [
        {
          text: 'Chapter 1 · FusionANNS',
          items: [
            { text: 'FusionANNS架构设计', link: '/more/chapter1/GPU加速检索-基于FusionANNS' }
          ]
        },
        {
          text: 'Chapter 2 · Meta-Chunking',
          items: [
            { text: '全新的文本切分策略', link: '/more/chapter2/Meta-Chunking：一种新的文本切分策略' }
          ]
        },
        {
          text: 'Chapter 3 · Limit',
          items: [
            { text: '基于嵌入检索的理论极限', link: '/more/chapter3/Limit基于嵌入检索的理论极限' },
            { text: '实践操作', link: '/more/chapter3/Meta_limit/code/startup' }
          ]
        },
        {
          text: 'Chapter 4 · RabitQ',
          items: [
            { text: 'RabitQ索引', link: '/more/chapter4/RabitQ：用于近似最近邻搜索的带理论误差界的高维向量量化' }
          ]
        },
        {
          text: 'Chapter 5 · 稀疏、稠密向量',
          items: [
            { text: '向量基础知识', link: '/more/chapter5/向量' }
          ]
        },
        {
          text: 'Chapter 6 · 聚类算法',
          items: [
            { text: '两种聚类算法', link: '/more/chapter6/聚类算法介绍' },
            { text: 'K-means聚类详解', link: '/more/chapter6/K-mean算法详解' }
          ]
        },
        {
          text: 'Chapter 7 · Milvus 架构',
          items: [
            { text: 'Milvus底层架构设计详解', link: '/more/chapter7/Milvus底层架构详解' }
          ]
        },
        {
          text: 'Chapter 8 · Milvus 索引',
          items: [
            { text: 'Milvus 索引详解', link: '/more/chapter8/Milvus索引详解' }
          ]
        },
      ],
      '/projects/': [
        {
          text: '实战项目',
          items: [
            { text: '项目概览', link: '/projects/' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/easy-vectordb' }
    ],

    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2025 datawhale'
    },

    search: {
      provider: 'local'
    },

    editLink: {
      pattern: 'https://github.com/datawhalechina/easy-vectordb/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    }
  }
})
