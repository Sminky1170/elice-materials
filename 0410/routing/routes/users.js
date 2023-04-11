const { Router } = require("express");

const router = Router();

router.get("/board", (req, res) => {
  res.send("board");
});

router.get("/profile", (req, res) => {
  res.send("profile");
});

router.get("/private", (req, res) => {
  res.send("private");
});

module.exports = router;
