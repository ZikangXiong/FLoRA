import { defineUserConfig } from 'vuepress'
import { defaultTheme } from '@vuepress/theme-default'
import { viteBundler } from '@vuepress/bundler-vite'
import { path } from '@vuepress/utils'
import { registerComponentsPlugin } from '@vuepress/plugin-register-components'
import markdownItKatex from 'markdown-it-katex'


export default defineUserConfig({
    base: '/FLoRA/',
    bundler: viteBundler(),
    theme: defaultTheme({
        navbar: [
            {
                text: 'Home',
                link: '/',
            },
            {
                text: 'Supplementary Material',
                link: '/subpages/supplementary',
            }
        ],
        sidebar: [],
        lastUpdated: false,
        contributors: false,
        colorMode: 'auto',
        colorModeSwitch: true,
    }),
    plugins: [
        registerComponentsPlugin({
            componentsDir: path.resolve(__dirname, './components'),
        }),
    ],
    extendsMarkdown: (md) => {
        md.use(markdownItKatex)
    },
    head: [
        ['link', {
            rel: 'stylesheet',
            href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.15.2/katex.min.css'
        }]
    ],
    title: 'FLoRA',
    description: 'A Framework for Learning Scoring Rules in Autonomous Driving Planning Systems',
    public: path.resolve(__dirname, './public'),
})