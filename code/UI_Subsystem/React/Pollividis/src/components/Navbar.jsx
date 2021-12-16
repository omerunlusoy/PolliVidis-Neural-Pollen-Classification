import {
  alpha,
  AppBar,
  Avatar,
  Badge, Button,
  InputBase,
  makeStyles,
  Toolbar,
  Typography,
} from "@material-ui/core";
import React, { useState } from "react";
import TemporaryDrawer from "./Drawer";

const useStyles = makeStyles((theme) => ({
  toolbar: {
    display: "flex",
    justifyContent: "space-between",
  },
  logoLg: {
    display: "none",
    [theme.breakpoints.up("sm")]: {
      display: "block",
    },
  },
  logoSm: {
    display: "block",
    [theme.breakpoints.up("sm")]: {
      display: "none",
    },
  },
  icons: {
    alignItems: "center",
    display: (props) => (props.open ? "none" : "flex"),
  },
  badge: {
    marginRight: theme.spacing(2),
  },
}));

const Navbar = () => {
  const [open, setOpen] = useState(false);
  const classes = useStyles({ open });
  return (
    <AppBar position="fixed" style={{backgroundColor:'black'}}>
      <Toolbar className={classes.toolbar}>
        <Typography variant="h6" className={classes.logoLg}>
         POLLIVIDIS
        </Typography>
        <div className={classes.icons}>
          <Badge className={classes.badge}>
            <TemporaryDrawer/>
          </Badge>
        </div>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
