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

    nav: [
      { text: '首页', link: '/' },
      { text: '向量基础', link: '/base/项目介绍' },
      { text: 'Annoy 教程', link: '/Annoy/chapter1/Annoy介绍' },
      { text: 'Milvus 教程', link: '/Milvus/chapter1/Milvus 介绍' },
      { text: '实战项目', link: '/projects/' }
    ],

    sidebar: {
      '/base/': [
        {
          text: 'Chapter 1 · 基础概念',
          items: [
            { text: '1.为何需要向量数据库', link: '/base/项目介绍' },
            { text: '2.向量嵌入基础', link: '/base/向量嵌入' }
          ]
        }
      ],

      '/Annoy/': [
        {
          text: 'Chapter 1 · 基础概念',
          items: [
            { text: 'Annoy 介绍', link: '/Annoy/chapter1/Annoy介绍' },
            { text: '基础API使用', link: '/Annoy/chapter1/基础API使用' }
          ]
        }
      ],

      '/Milvus/': [
        {
          text: 'Chapter 1 · 基础概念',
          items: [
            { text: 'Milvus 介绍', link: '/Milvus/chapter1/Milvus 介绍' },
            { text: 'milvus 索引介绍', link: '/Milvus/chapter1/milvus 索引介绍' },
            { text: '聚类算法介绍', link: '/Milvus/chapter1/聚类算法介绍' }
          ]
        },
        {
          text: 'Chapter 2 · 基础部署',
          items: [
            { text: 'Milvus Lite 部署与应用', link: '/Milvus/chapter2/Milvus Lite部署与应用' },
            { text: 'Milvus Standalone 部署', link: '/Milvus/chapter2/Milvus Standalone部署' },
            { text: 'MinerU 部署教程', link: '/Milvus/chapter2/MinerU部署教程' }
          ]
        },
        {
          text: 'Chapter 3 · 实战进阶',
          items: [
            { text: 'milvus 文本嵌入实战', link: '/Milvus/chapter3/milvus 文本嵌入实战' },
            { text: 'milvus pdf 嵌入实战', link: '/Milvus/chapter3/milvus pdf 嵌入实战' },
            { text: 'milvus pdf 多模型嵌入实战', link: '/Milvus/chapter3/milvus pdf 多模型嵌入实战' },
            { text: 'milvus 数据切分总结', link: '/Milvus/chapter3/milvus 数据切分总结' }
          ]
        },
        {
          text: 'Chapter 4 · 优化与实践',
          items: [
            { text: 'milvus 存储优化', link: '/Milvus/chapter4/milvus 存储优化' },
            { text: 'GPU 加速检索 - 基于 FusionANNS', link: '/Milvus/chapter4/GPU加速检索-基于FusionANNS' },
            { text: '向量', link: '/Milvus/chapter4/向量/向量.md' },
            { text: 'Meta-Chunking：一种新的文本切分策略', link: '/Milvus/chapter4/Meta-Chunking：一种新的文本切分策略' }
          ]
        }
      ],
      // '/Faiss/': [
      //   {
      //     text: 'Faiss 教程',
      //     items: [
      //       { text: 'Faiss 核心原理与架构', link: '/Faiss/1.1Faiss核心原理与架构' }
      //     ]
      //   }
      // ],
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
