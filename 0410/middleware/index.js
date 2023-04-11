const express = require("express");
const usersRouter = require("./routes/users");
const app = express();

app.use("/users", logRequest, validateUser, usersRouter);

const logRequest = (req, res, next) => {
  console.log(req.method);
  // 로직
  next();
};

const validateUser = (req, res, next) => {
  if (req.params.name === "min") {
    // 통과
    next();
  }
  // 실패
};

app.get("/", (req, res) => {
  res.send("");
});

app.listen(8080);
