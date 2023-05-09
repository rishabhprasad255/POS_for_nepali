let nov_dec_2021 = require("./static/nov_dec_2021");
const fs = require("fs");
const { error } = require("console");

const temp = nov_dec_2021.array;
const data = JSON.stringify(temp);
fs.writeFile("E:/flask-mini/static/subname/nov_dec_2021.json", data, error);
