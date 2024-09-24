<template>
    <div class="bibtex">
        <pre><code>{{ bibtexContent }}</code></pre>
    </div>
    <br>
</template>

<script setup>
const bibtexContent = `@inproceedings{xiong24flora,
    title={FLoRA: A Framework for Learning Scoring Rules in Autonomous Driving Planning Systems},
    author={Zikang Xiong, Joe Eappen, and Suresh Jagannathan},
    booktitle={Preprint},
    year={2024},
  }`
</script>

<style scoped>
.bibtex {
    background-color: #f4f4f4;
    padding: 1rem;
    border-radius: 4px;
    overflow-x: auto;
}
</style>