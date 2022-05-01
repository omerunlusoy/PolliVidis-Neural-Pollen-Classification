import {Card, CardActionArea, CardContent, CardMedia, Container, Grid, makeStyles, Typography} from "@material-ui/core";
import Post from "./Post";
import Navbar from "./Navbar";
import Leftbar from "./Leftbar";
import Add from "./Add";
import SampleImagePreviewCard from "./SampleImagePreviewCard";
import React, {useEffect} from "react";
import ImageCard from "./ImageCard";
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemAvatar from '@mui/material/ListItemAvatar';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Avatar from '@mui/material/Avatar';
import IconButton from '@mui/material/IconButton';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import FolderIcon from '@mui/icons-material/Folder';
import DeleteIcon from '@mui/icons-material/Delete';
import {Button} from "@mui/material";

const useStyles = makeStyles((theme) => ({
  container: {
    paddingTop: theme.spacing(10),
  },
}));


const AboutUs = () => {

    const id_s = JSON.parse(sessionStorage.getItem('academic_id'));
    let id = id_s.academic_id

    const Demo = styled('div')(({ theme }) => ({
        backgroundColor: theme.palette.background.paper,
    }));
  const classes = useStyles();
    const [dense, setDense] = React.useState(false);
    const [secondary, setSecondary] = React.useState(false);
    const [markers, setMarkers] = React.useState([]);

    useEffect(() => {
        fetch(`http://localhost:8000/api/get_samples_of_academic/${id}/`)
            .then((data) => data.json())
            .then((data) => setMarkers(data))
            .then((data) => console.log(data))
    },[]);


  return (
      <div>
          <Navbar />
          <Grid container>
              <Grid item sm={7} xs={10}>
                  <Container className={classes.container}>
                      <Typography style={{marginBottom:10}}variant="h3" component="p">
                          Previous Analyses
                      </Typography>
                                  <Demo>
                                      <List dense={dense}>
                                          {markers.map((marker) => (
                                              <ListItem
                                                  secondaryAction={
                                                      <Button type="submit" variant="contained" style={{backgroundColor:'#A6232A', color:'white'}} >
                                                          View
                                                      </Button>
                                                  }
                                              >
                                                  <ListItemAvatar>
                                                      <Avatar>
                                                          <FolderIcon />
                                                      </Avatar>
                                                  </ListItemAvatar>
                                                  <ListItemText
                                                      primary={marker.date}
                                                      secondary={secondary ? 'Secondary text' : null}
                                                  />
                                              </ListItem>
                                          ))}
                                      </List>
                                  </Demo>

                  </Container>
              </Grid>
              <Grid item sm={3} className={classes.right}>

              </Grid>
          </Grid>
      </div>
  );
};

export default AboutUs;
