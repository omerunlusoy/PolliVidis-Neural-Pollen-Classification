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
import RightDrawer from "./Drawer";
import {Link} from "react-router-dom";

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
        <Link style={{ textDecoration: 'none' }} to="/map"><Button variant = 'text' disableRipple={true} style={{
          color: 'white',
          fontSize : 'large'
        }} className={classes.text}>POLLIVIDIS</Button></Link>

        <div className={classes.icons}>
          <Badge className={classes.badge}>
            <RightDrawer/>
          </Badge>
        </div>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
