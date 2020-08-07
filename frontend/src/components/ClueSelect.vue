<template>
  <div class="clue-select-ctr">
    <div class="clues-ctr">
      <div class='clue-select-text'>What's the best clue?</div>
      <div class='clue' v-for="clue in clues" :key="clue.id">
        <label :for="clue.word">{{clue.word}}</label>
        <input type="radio" v-model="selectedClue" :name="clue.word" :value="clue.word">
      </div>
      <div class='clue all_bad'>
        <label for="all_bad">I don't like any of these.</label>
        <input type="radio" v-model="selectedClue" name="all_bad" value="no_good_clues">
      </div>

    </div>
    <div class='clue-text-ctr'>
      <div class="clue-text">{{selectedClue}}</div>
      <div class='clue-helper-text'>for {{numTopFriends}}</div>
    </div>
    <button class="submit-btn" :disabled="!selectedClue" @click.prevent="submit">Submit</button>
  </div>
</template>

<script>
import {mapGetters, mapActions} from 'vuex'
export default {
  name: 'ClueSelect',
  data() {
    return {
      selectedClue: ''
    }
  },
  methods: {
    ...mapActions(['submitReview']),
    submit() {
      this.submitReview({selectedClue: this.selectedClue})
    }
  },
  computed: {
    ...mapGetters(['clues', 'numTopFriends'])
  },
}
</script>

<style>
  .clue-select-ctr {
    position: relative;
    width: 600px;
    height: 250px;
    margin: 1rem auto 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .clue-select-text {
    font-weight: 400;
    color: grey;
    margin-bottom: .5rem;
  }
  .clues-ctr {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .clue {
    width: 200px;
    display: flex;
    margin-bottom: .2rem;
    padding: .4rem;
    border: 1px solid grey;
    justify-content: space-between;
    align-items: center;
  }
  .all_bad {
    background-color: rgb(231, 231, 231);
  }
  .clue-text-ctr {
    position: relative;
    display: flex;
  }
  .clue-text {
    font-size: 1.5rem;
    width: 250px;
    border-bottom: 5px solid black;
  }
  .clue-helper-text {
    font-size: 2.5rem;
    font-weight: 700;
    margin-left: 1rem;
  }
  .submit-btn {
    position: absolute;
    right: 0;
    bottom: 0;
  }
</style>