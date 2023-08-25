const express = require("express");
const app = express();

app.use(express.static("dist"));
app.listen(443);

console.log(`server running on port ${443}`)