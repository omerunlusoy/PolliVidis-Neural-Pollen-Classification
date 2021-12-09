import {Button, Container, makeStyles, Typography} from "@material-ui/core";
import {
  Bookmark,
  List,
  ExitToApp,
  Home,
  Person,
  PhotoCamera,
  PlayCircleOutline,
  Settings,
  Storefront,
  TabletMac,
} from "@material-ui/icons";
import FeedbackIcon from '@material-ui/icons/Feedback';
import GroupIcon from '@material-ui/icons/Group';
import GetAppIcon from '@material-ui/icons/GetApp';
import HelpIcon from '@material-ui/icons/Help';
import FolderIcon from '@material-ui/icons/Folder';
import React, {useState} from "react";
import {Navigate} from "react-router-dom";
import {  Link } from "react-router-dom";

const useStyles = makeStyles((theme) => ({
  container: {
    height: "100vh",
    color: "white",
    paddingTop: theme.spacing(10),
    backgroundColor: theme.palette.primary.main,
    position: "sticky",
    top: 0,
    [theme.breakpoints.up("sm")]: {
      backgroundColor: "white",
      color: "#555",
      border: "1px solid #ece7e7",
    },
  },
  item: {
    display: "flex",
    alignItems: "center",
    marginBottom: theme.spacing(4),
    [theme.breakpoints.up("sm")]: {
      marginBottom: theme.spacing(3),
      cursor: "pointer",
    },
  },
  icon: {
    marginRight: theme.spacing(1),
    [theme.breakpoints.up("sm")]: {
      fontSize: "18px",
    },
  },
  text: {
    fontWeight: 500,
    [theme.breakpoints.down("sm")]: {
      display: "none",
    },
  },
}));

const Leftbar = (props) => {
  const classes = useStyles();

  return (
    <Container className={classes.container}>
      <div>
        <div className={classes.item}>
          <Home className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/login"><Button style={{textAlign:"left"}} className={classes.text}>Academic Login</Button></Link>
        </div>
        <div className={classes.item}>
          <Person className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/profile"><Button style={{textAlign:"left"}} className={classes.text}> Profile</Button></Link>
        </div>
        <div className={classes.item}>
          <FolderIcon className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/"><Button style={{textAlign:"left"}} className={classes.text}>Previous Analysis</Button></Link>
        </div>
        <div className={classes.item}>
          <FolderIcon className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/map"><Button style={{textAlign:"left"}} className={classes.text}>Pollen Map</Button></Link>
        </div>
        <div className={classes.item}>
          <GetAppIcon className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/"><Button style={{textAlign:"left"}} className={classes.text}>Download Dataset</Button></Link>
        </div>
        <div className={classes.item}>
          <GroupIcon className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/about-us"><Button style={{textAlign:"left"}} className={classes.text}>About Us</Button></Link>
        </div>
        <div className={classes.item}>
          <FeedbackIcon className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/"><Button style={{textAlign:"left"}} className={classes.text}>Send Feedback</Button></Link>
        </div>
        <div className={classes.item}>
          <HelpIcon className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/"><Button style={{textAlign:"left"}} className={classes.text}>How PolliVidis Works</Button></Link>
        </div>
        <div className={classes.item}>
          <ExitToApp className={classes.icon} />
          <Link style={{ textDecoration: 'none'}} to="/"><Button style={{textAlign:"left"}} className={classes.text}>Logout</Button></Link>
        </div>
      </div>
    </Container>
  );
};

export default Leftbar;
