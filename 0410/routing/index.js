const express = require("express");
const usersRouter = require("./routes/users");

const app = express();

app.get("/", (req, res) => {
  res.send("hello");
});

app.use("/users", usersRouter);

app.listen(8080);
