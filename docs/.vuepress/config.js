import { defineUserConfig } from 'vuepress'
import { viteBundler } from '@vuepress/bundler-vite'
import { hopeTheme } from "vuepress-theme-hope"
import { path } from '@vuepress/utils'


export default defineUserConfig({
    base: '/FLoRA/',
    bundler: viteBundler(),
    theme: hopeTheme({
        navbar: [
            {
                text: 'Home',
                link: '/',
            },
            {
                text: 'Scenarios',
                link: '/scenarios/',
            }
        ],
        sidebar: [

        ],
        pageLayout: {
            wide: true
        },
        plugins: {
            readingTime: false,
            lastUpdated: false,
        },
    }),
    // other config options
    title: 'FLoRA',
    description: 'A Framework for Learning Scoring Rules in Autonomous Driving Planning Systems',
    public: path.resolve(__dirname, './public'),
})