<template>
    <div class="bibtex">
        <pre><code>{{ bibtexContent }}</code></pre>
    </div>
    <br>
</template>

<script setup>
const bibtexContent = `@misc{xiong24flora,
  title={FLoRA: A Framework for Learning Scoring Rules in Autonomous Driving Planning Systems},
  author={Xiong, Zikang and Eappen, Joe and Jagannathan, Suresh},
  year={2024},
  primaryClass={cs.RO},
  note={Preprint}
}`
</script>

<style scoped>
.bibtex {
    background-color: #f4f4f4;
    padding: 1rem;
    border-radius: 4px;
    overflow-x: auto;
    color: #333;
    /* Dark text color for light mode */
}

@media (prefers-color-scheme: dark) {
    .bibtex {
        background-color: #2a2a2a;
        color: #e0e0e0;
        /* Light text color for dark mode */
    }
}
</style>