<template>
  <div class="status-ctr">
    <div>You have reviewed {{reviewCount}} {{gameString}}.</div>
  </div>
</template>

<script>
export default {
  name: 'Status',
  computed: {
    reviewCount() {
      const reviewCount = parseInt(localStorage.getItem('codenamesReviewCount'))
      return reviewCount ? reviewCount : 0
    },
    gameString() {
      return this.reviewCount == 1 ? 'game' : 'games'
    }
  },
}
</script>

<style>
  .status-ctr {
    position: absolute;
    left: 1%;
    bottom: 1%;
  }
</style>