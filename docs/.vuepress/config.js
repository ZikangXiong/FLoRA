import { defineUserConfig } from 'vuepress'
import { defaultTheme } from '@vuepress/theme-default'
import { viteBundler } from '@vuepress/bundler-vite'
import { path } from '@vuepress/utils'
import { registerComponentsPlugin } from '@vuepress/plugin-register-components'


export default defineUserConfig({
    base: '/FLoRA/',
    bundler: viteBundler(),
    theme: defaultTheme({
        navbar: [
            {
                text: 'Home',
                link: '/',
            }
        ],
        sidebar: [],
        // disable updata time
        lastUpdated: false,
        // disable contributors
        contributors: false,
        // disable dark mode
        darkMode: false,
    }),
    plugins: [
        registerComponentsPlugin({
            componentsDir: path.resolve(__dirname, './components'),
        }),
    ],
    title: 'FLoRA',
    description: 'A Framework for Learning Scoring Rules in Autonomous Driving Planning Systems',
    public: path.resolve(__dirname, './public'),
})