const express = require("express");
const usersRouter = require("./routes/users");

const app = express();

const middleware = (req, res, next) => {
  console.log("middleware 거쳐감");
  next();
};

app.use(middleware);

app.get("/", (req, res) => {
  res.send("Home");
});

app.use("/users", usersRouter);

app.listen(4000);
