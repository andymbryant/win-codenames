<template>
  <div :class="cardContainerClass">
    <div :class="cardTextClass">{{word}}</div>
    <font-awesome-icon class="top-friend-icon" v-if="isTopFriend" icon="check-circle"/>
  </div>
</template>

<script>
export default {
  name: "Card",
  props: {
    word: String,
    type: String,
    isTopFriend: Boolean
  },
  computed: {
    cardContainerClass() {
      return {
        'card-ctr': true,
        'assassin-card-ctr': this.type === 'assassin'
      }
    },
    cardTextClass() {
      return {
        'card-text': true,
        'friend-card': this.type === 'friend',
        'foe-card': this.type === 'foe',
        'neutral-card': this.type === 'neutral',
        'assassin-card': this.type === 'assassin',
      }
    }
  }
}
</script>

<style>
  .card-ctr {
    width: 133px;
    height: 90px;
    line-height: 90px;
    text-align: center;
    position: relative;
    background-color: #e8e8e8;
  }
  .card-text {
    display: inline-block;
    vertical-align: middle;
    line-height: normal;
    text-transform: uppercase;
    font-weight: 700;
  }
  .friend-card {
    color: #d13030;
  }

  .foe-card {
    color: #4183cc;
  }

  .neutral-card, .assassin-card {
    color: black;
  }
  .assassin-card-ctr {
    background-color: rgb(182, 182, 182);
  }

  .top-friend-icon {
    position: absolute;
    right: 10px;
    bottom: 10px;
    color: green;
  }

</style>