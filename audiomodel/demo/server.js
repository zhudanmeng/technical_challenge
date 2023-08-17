const express = require("express");
const app = express();

app.use(express.static("dist"));
app.listen(process.argv[3]);

console.log(`server running on port ${process.argv[3]}`)