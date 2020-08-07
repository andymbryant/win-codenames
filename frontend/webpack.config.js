module: {
  rules: [
      //other stuff
      { test: /\.(png|woff|woff2|eot|ttf|svg)$/, use: 'url-loader?limit=25000' }
  ]
}