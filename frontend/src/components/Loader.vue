<template>
  <div class="loader-ctr">
    <div class='loader'></div>
  </div>
</template>

<script>
export default {
  name: "Loader"
}
</script>

<style>

  .loader-ctr {
    width: 100%;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* https://www.w3schools.com/howto/tryit.asp?filename=tryhow_css_loader */
  .loader {
    /* margin: 0 auto; */
    border: 16px solid #f3f3f3;
    border-radius: 50%;
    border-top: 16px solid green;
    width: 120px;
    height: 120px;
    -webkit-animation: spin 2s linear infinite; /* Safari */
    animation: spin 2s linear infinite;
  }

  /* Safari */
  @-webkit-keyframes spin {
    0% { -webkit-transform: rotate(0deg); }
    100% { -webkit-transform: rotate(360deg); }
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

</style>