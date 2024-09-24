<template>
    <div class="video-container">
        <video width="100%" controls>
            <source :src="$withBase(src)" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
</template>

<script setup>
defineProps({
    src: {
        type: String,
        required: true
    }
})
</script>

<style scoped>
.video-container {
    max-width: 1100px;
    margin: 0 auto;
}
</style>