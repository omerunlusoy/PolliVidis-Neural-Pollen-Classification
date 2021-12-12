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
import {Cancel, Home, Mail, Notifications, Search} from "@material-ui/icons";
import React, { useState } from "react";
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
    <AppBar position="fixed">
      <Toolbar className={classes.toolbar}>
        <Typography variant="h6" className={classes.logoLg}>
         POLLIVIDIS
        </Typography>
        <div className={classes.icons}>
          <Badge  color="secondary" className={classes.badge}>
            <Link style={{ textDecoration:'none'}} to="/"><Button style={{ color: '#FFF'}}>Analyze Sample</Button></Link>
          </Badge>
        </div>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
