<template>
    <div class="button-container">
        <a v-for="button in buttons" :key="button.text" :href="button.href" :class="['custom-button', button.class]">
            {{ button.text }}
        </a>
    </div>
</template>

<script setup>
const buttons = [
    { text: 'PDF', href: 'https://xiong.zikang.me/FLoRA/draft-v1.pdf', class: 'pdf-button' },
    { text: 'CODE', href: 'https://github.com/ZikangXiong/FLoRA', class: 'code-button' },
    { text: 'BIB', href: '#citation', class: 'bib-button' }
]
</script>

<style scoped>
.button-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin: 20px 0;
}

.custom-button {
    display: inline-block;
    padding: 10px 20px;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-weight: bold;
    transition: background-color 0.3s;
}

.pdf-button {
    background-color: #dc3545;
    /* Bootstrap red */
}

.pdf-button:hover {
    background-color: #c82333;
    /* Darker shade for hover effect */
}

.code-button {
    background-color: #0d1117;
    /* GitHub dark */
}

.code-button:hover {
    background-color: #161b22;
    /* Slightly lighter shade for hover effect */
}

.bib-button {
    background-color: #008080;
    /* LaTeX logo color (teal) */
}

.bib-button:hover {
    background-color: #006666;
    /* Darker shade for hover effect */
}

/* Remove VuePress external link icon and related styles */
:deep(.custom-button) {
    background-image: none !important;
    padding-right: 20px !important;
}

:deep(.custom-button::after) {
    display: none !important;
}
</style>