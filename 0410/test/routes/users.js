const { Router } = require("express");
const data = require("../user.json");

const router = Router();

router.get("/:name", (req, res) => {
  const userName = req.params.name;
  if (data[userName]) {
    res.send(String(data[userName].id));
  } else {
    res.status(404).end();
  }
});

module.exports = router;
