import {
  Button,
  Card,
  CardActionArea,
  CardActions,
  CardContent,
  CardMedia,
  makeStyles,
  Typography,
} from "@material-ui/core";

const useStyles = makeStyles((theme) => ({
  card: {
    marginTop: theme.spacing(10),
    marginBottom: theme.spacing(5),
  },
  media: {
    height: 250,
    [theme.breakpoints.down("sm")]: {
      height: 150,
    },
  },
}));

const SampleImagePreviewCard = ({ img}) => {
  const classes = useStyles();
  return (
      <Card className={classes.card}>
        <CardActionArea>
          <CardMedia className={classes.media} image={img}  />
          <CardContent>
            <Typography variant="body2" color="textSecondary" component="p">
              Sample Image Preview
            </Typography>
          </CardContent>
        </CardActionArea>
      </Card>
  );
};

export default SampleImagePreviewCard;
