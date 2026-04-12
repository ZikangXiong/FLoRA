<template>
    <div class="demo-gallery">
        <div v-if="loading">Loading...</div>
        <div v-else>
            <div class="dropdown">
                <select v-model="activeDemo">
                    <option v-for="(label, value) in demos" :key="value" :value="value">
                        {{ label }}
                    </option>
                </select>
            </div>

            <ul class="scenario-description">
                <li>{{ scenarioDescriptions[activeDemo] }}</li>
            </ul>

            <div class="video-gallery">
                <div v-for="i in 3" :key="i" class="video-item">
                    <video width="300" controls :key="`video-${activeDemo}-${i}`" ref="videoRefs">
                        <source :src="getVideoSrc(activeDemo, i)" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'

const activeDemo = ref('Change Lane')
const scenarioData = ref(null)
const loading = ref(true)
const videoRefs = ref([])

const demos = {
    'Change Lane': 'Change Lane',
    'Following': 'Following',
    'Near Static': 'Near Static',
    'Near VRU': 'Near VRU',
    'Turn': 'Turn',
    'Stopping': 'Stopping',
    'Starting': 'Starting',
    'Stationary': 'Stationary',
    'Traversing': 'Traversing'
}

const scenarioDescriptions = {
  'Change Lane': 'Vehicle executes a lane change maneuver between adjacent lanes',
  'Following': 'Vehicle maintains a safe following distance behind a leading vehicle',
  'Near Static': 'Vehicle navigates near static obstacles like parked cars',
  'Near VRU': 'Vehicle is near Vulnerable Road Users (VRUs) like pedestrians or cyclists',
  'Turn': 'Vehicle performs turning maneuvers at intersections',
  'Stopping': 'Vehicle comes to a controlled stop',
  'Starting': 'Vehicle initiates motion from a stationary position',
  'Stationary': 'Vehicle maintains a stable stationary position',
  'Traversing': 'Vehicle traverses through intersections or complex road sections'
};

const readScenarioData = async () => {
    try {
        const response = await fetch('/FLoRA/scenarios/videos/classified_scenarios.json')
        scenarioData.value = await response.json()
        loading.value = false
    } catch (error) {
        loading.value = false
    }
}

onMounted(readScenarioData)

const getVideoSrc = (demo, index) => {
    if (scenarioData.value && scenarioData.value[demo]) {
        const videos = scenarioData.value[demo]
        if (index <= videos.length) {
            const videoPath = videos[index - 1]
            const formattedPath = (videoPath).replace(/\/+/g, '/')
            const baseUrl = import.meta.env.BASE_URL || '';
            const formattedPathWithBase = baseUrl + formattedPath;
            return formattedPathWithBase;
        }
    }
    return ''
}

watch(activeDemo, (newDemo) => {
    nextTick(() => {
        videoRefs.value.forEach((video, index) => {
            if (video) {
                const newSrc = getVideoSrc(newDemo, index + 1)
                video.src = newSrc
                video.load()
            }
        })
    })
})

</script>

<style scoped>
.demo-gallery {
    margin-top: 2rem;
}

.dropdown {
    margin-bottom: 1rem;
}

select {
    padding: 0.5rem;
    font-size: 1rem;
    border-radius: 4px;
}

.video-gallery {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
}

.video-item {
    margin: 10px;
}
</style>