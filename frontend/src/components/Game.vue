<template>
  <Loader v-if="isAppLoading"></Loader>
  <div v-else class='game-ctr'>
    <div class='reviewer-ctr'>
      <label class="reviewer-label" for="reviewerName">What's your name?</label>
      <input type="text" name="reviewerName" placeholder="e.g. andyb, lindseym..." v-model="reviewerName">
    </div>
    <div class='title-ctr'>
      <div class='game-title'>You are the Spymaster for the <span class="red-text">Red Team</span>.</div>
      <div class='game-subtitle'>You want your teammates to guess the words with the <span class="green-text">green</span> checkmarks.</div>
    </div>
    <span v-if="!loading">
      <Board/>
      <ClueSelect/>
      <Status/>
      <div class='copyright'>© 2020 Andy Bryant</div>
    </span>
  </div>
</template>

<script>
import Loader from '@/components/Loader'
import Board from '@/components/Board'
import Status from '@/components/Status'
import ClueSelect from '@/components/ClueSelect'
import {mapActions, mapGetters} from 'vuex'

export default {
  name: 'Game',
  components: {
    Loader,
    Board,
    ClueSelect,
    Status
  },
  data() {
    return {
      loading: false
    }
  },
  methods: {
    ...mapActions(['fetchGameData', 'fetchClueData']),
  },
  computed: {
    ...mapGetters(['clues', 'isAppLoading']),
    reviewerName: {
      get: function() {
        let name = localStorage.getItem('codenamesReviewerName')
        return name ? name : ''
      },
      set: function(newValue) {
        localStorage.setItem('codenamesReviewerName', newValue)
      }
    }
  },
  created() {
    this.loading = true
  },
  mounted() {
    this.fetchGameData().then(() => this.loading = false)
  }
}
</script>

<style>
  .game-ctr {
    width: 100%;
    text-align: center;
  }
  .reviewer-ctr {
    position: absolute;
    left: 1%;
    top: 1%;
  }
  .reviewer-label {
    margin-right: 1rem;
  }
  .title-ctr {
    margin: 1rem 0;
    display: flex;
    flex-direction: column;
  }
  .game-title {
    font-size: 1.5rem;
    height: 40px;
    margin-top: 1rem;
    font-weight: 500;
  }
  .red-text {
    color: #d13030;
    font-weight: 700;
  }
  .green-text {
    color: green;
    font-weight: 700;
  }
  .main-ctr {
    display: flex;
    flex-direction: column;
  }
  .copyright {
    position: absolute;
    right: 1%;
    bottom: 1%;
    color: rgb(172, 172, 172);
  }
</style>