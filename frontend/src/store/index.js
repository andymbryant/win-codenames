import Vuex from 'vuex';
import Vue from 'vue';
import game from './modules/game.js';

Vue.use(Vuex);

export default new Vuex.Store({
  modules: {
    game
  }
});
