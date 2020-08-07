<template>
  <div class="board-ctr">
    <Card
      v-for="word in words"
      :word="word.word"
      :type="word.type"
      :isTopFriend="word.is_top_friend"
      :key="word.id"
      >
    </Card>
  </div>
</template>

<script>
import Card from '@/components/Card.vue'
import {mapGetters} from 'vuex'
export default {
  name: 'Board',
  components: {
    Card
  },
  computed: {
    ...mapGetters(['words'])
  },
}
</script>

<style>
  .board-ctr {
    margin: 0 auto;
    width: 700px;
    height: 500px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
  }
</style>