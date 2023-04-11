const { Router } = require("express");

const router = Router();

router.get("/board", (req, res) => {
  // 로직
  res.send("board");
});

router.get("/profile", (req, res) => {
  // 로직
  res.send("profile");
});

router.get("/private", (req, res) => {
  // 로직
  res.send("private");
});

module.exports = router;
