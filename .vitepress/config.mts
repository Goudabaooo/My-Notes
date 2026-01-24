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
      { text: 'Home', link: '/' },
      { text: 'Examples', link: '/markdown-examples' }
    ],

    sidebar: [
      {
        text: 'Examples',
        items: [
          { text: 'Markdown Examples', link: '/markdown-examples' },
          { text: 'Runtime API Examples', link: '/api-examples' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
    
  }
})
