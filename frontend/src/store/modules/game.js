import {shuffle} from '@/helpers.js'
import axios from 'axios'

const ROOT_URL = process.env.VUE_APP_FLASK

const SHUFFLE_CLUES = true
const SHUFFLE_WORDS = true

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function getDefaultState() {
  return {
    isAppLoading: false,
    gameId: "",
    processConfig: {},
    words: [],
    clues: []
  }
}

const state = getDefaultState()

const getters = {
  isAppLoading: state => state.isAppLoading,
  gameId: state => state.gameId,
  words: state => state.words,
  clues: state => state.clues,
  numTopFriends: state => state.words.filter(w => w.is_top_friend && w.type === 'friend').length
}

const actions = {
  async fetchGameData({commit}, payload) {
    commit('toggleAppLoading')
    let url = `${ROOT_URL}/games/`
    if (payload && Object.prototype.hasOwnProperty.call(payload, 'id')) {
      url = url + payload.id + '/'
    }
    try {
      const game = await axios.get(url).then(res => res.data[0])
      await commit('loadGameData', game)
    } catch(error) {
      console.error(error)
    }
    await sleep(600)
    commit('toggleAppLoading')
  },
  async submitReview({getters, dispatch}, {selectedClue}) {
    let url = `${ROOT_URL}/create-review/`
    const reviewerName = localStorage.getItem('codenamesReviewerName')
    const data = {
      reviewer: reviewerName ? reviewerName : null,
      human_selected: selectedClue,
      game_id: getters.gameId,
      num_displayed: getters.clues.length,
      order_displayed: getters.clues.map(c => c.word),
      shuffled_clues: SHUFFLE_CLUES
    }
    try {
      await axios.post(url, data)
      await dispatch('incrementReviewCount')

    } catch(error) {
      console.error(error)
    }
    await dispatch('fetchGameData')
  },
  incrementReviewCount() {
    let reviewCount = parseInt(localStorage.getItem('codenamesReviewCount'))
    if (!reviewCount) {
      reviewCount = 0
    }
    reviewCount++
    localStorage.setItem('codenamesReviewCount', reviewCount.toString())
  }
}

const mutations = {
  toggleAppLoading(state) {
    state.isAppLoading = !state.isAppLoading
  },
  loadGameData(state, payload) {
    const {words, id, clues, process_config} = payload
    state.words = SHUFFLE_WORDS ? shuffle(words) : words
    state.clues = SHUFFLE_CLUES ? shuffle(clues) : clues
    state.gameId = id
    state.processConfig = process_config
  }
};

export default {
  state,
  getters,
  actions,
  mutations
};