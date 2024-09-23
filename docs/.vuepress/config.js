import { defineUserConfig } from 'vuepress'
import { defaultTheme } from '@vuepress/theme-default'
import { viteBundler } from '@vuepress/bundler-vite'


export default defineUserConfig({
    base: '/FLoRA/',
    bundler: viteBundler(),
    theme: defaultTheme({
        // default theme config
        navbar: [
            {
                text: 'Home',
                link: '/',
            },
            // Add more navbar items as needed
        ],
        sidebar: [
            // Add sidebar items as needed
        ],
    }),
    // other config options
    title: 'Your Site Title',
    description: 'Your site description',
})