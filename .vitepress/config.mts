import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "My Awesome Project",
  description: "A VitePress Site",

  // 👇 2. 新增这块 markdown 配置代码
  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },
  // 👆 新增结束

  // 👇 复制这一段！添加这个 head 配置
  head: [
    ['meta', { name: 'referrer', content: 'no-referrer' }]
  ],
  // 👆 复制结束
  
  
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '主页', link: '/' },
      { text: '关于我', link: '/markdown-examples' }
    ],

    sidebar: [
      {
        text: '📅 近期聚焦',
        collapsed: false, // 默认展开
        items: [
          { text: '日程安排', link: '/Schedule_Howard' },
          { text: '组会记录 (2025-10-29)', link: '/组会20251029' },
          { text: '暑假学习情况', link: '/暑假学习情况' }
        ]
      },
      {
        text: '🎓 学术与课程',
        collapsed: false,
        items: [
          { text: '统计学习理论', link: '/StatisticalLearningTheory' },
          { text: '神经网络优化算法', link: '/神经网络优化算法模型' },
          { text: '机器学习课程论文', link: '/机器学习与数据挖掘课程论文' },
          { text: '毕业课题选择', link: '/毕业课题 选择' },
          { text: '正大杯项目', link: '/正大杯' }
        ]
      },
      {
        text: '🧩 随笔与测试',
        collapsed: true, // 默认折叠，不占地方
        items: [
          { text: '技术 Tips', link: '/tips' },
          { text: 'API 示例', link: '/api-examples' },
          { text: 'Markdown 示例', link: '/markdown-examples' },
          { text: '测试页面', link: '/这是一个测试' }
        ]
      }

    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Goudabaooo/My-Notes' }
    ]
    
  }
})
