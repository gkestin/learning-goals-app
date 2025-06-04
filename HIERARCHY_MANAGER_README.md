# Learning Goals Hierarchy Manager

The Learning Goals Hierarchy Manager allows you to upload, view, and manage hierarchical learning goals data from CSV files. This feature is designed to work with structured learning data that represents different levels of granularity or complexity.

## Features

- **CSV Upload**: Upload hierarchical learning goals from CSV files
- **Interactive Tree View**: Visualize the hierarchy with collapsible/expandable nodes
- **Editable Labels**: Click on node labels to rename them in real-time
- **Database Storage**: Save your hierarchy structure to the database
- **Level Analysis**: View learning goals grouped by their level patterns

## CSV File Format

Your CSV file should have the following structure:

```csv
learning_goal,level_0,level_1,level_2,level_3,level_4,level_5,level_6,level_7,level_8,level_9
Describe the shape of solution trajectories of the Lotka-Volterra model,5364,556,128,50,31,23,15,12,7,6
Analyze predator-prey dynamics using phase plane analysis,5364,556,128,50,31,23,15,12,8,4
```

### Column Requirements:

1. **learning_goal** (required): The text of the learning goal
2. **level_0 to level_9** (optional): Numeric values representing different hierarchy levels
   - Higher numbers typically represent higher levels in the hierarchy
   - Zero values are treated as "not present at this level"
   - Goals with similar level patterns are grouped together

## How to Use

### 1. Upload Your CSV File

1. Navigate to the "Hierarchy Manager" page from the main navigation
2. Drag and drop your CSV file into the upload area, or click "Browse Files"
3. The system will validate and process your file
4. Upon successful upload, the hierarchy will be automatically saved to the database

### 2. View and Navigate the Hierarchy

- **Expand/Collapse**: Click on any node header to expand or collapse its contents
- **Bulk Operations**: Use "Expand All" or "Collapse All" buttons for convenience
- **Level Patterns**: Each node shows its level pattern (e.g., "0:5364 → 1:556 → 2:128")

### 3. Edit Node Labels

1. Click on any node label (the bold text like "Group_1")
2. The label becomes editable - type your new name
3. Press Enter to save, or Escape to cancel
4. Modified labels are marked for saving

### 4. Save Changes

- Click the "Save Hierarchy to Database" button to persist all your changes
- The system will save both the structure and any label modifications
- A success message confirms the save operation

## Data Grouping Logic

The system automatically groups learning goals based on their level patterns:

- Goals with identical non-zero level values are grouped together
- Each group becomes a node in the hierarchy
- Groups are automatically labeled as "Group_1", "Group_2", etc.
- You can rename these groups to meaningful names

## Example Use Cases

### Educational Curriculum Design
- Upload learning objectives with different complexity levels
- Group related concepts by cognitive difficulty
- Create clear hierarchical progressions

### Competency Mapping
- Map skills to different proficiency levels
- Visualize learning pathways
- Track progression requirements

### Assessment Planning
- Organize questions by difficulty and topic
- Create structured test blueprints
- Ensure coverage across all levels

## Technical Notes

- Maximum file size: Depends on your server configuration
- Supported format: CSV files only
- Character encoding: UTF-8 recommended
- The system handles missing values gracefully (treated as 0)

## Sample Data

A sample CSV file (`sample_learning_goals_hierarchy.csv`) is included with example data about dynamical systems and differential equations learning goals.

## Troubleshooting

### Upload Issues
- **"First column must be 'learning_goal'"**: Ensure your first column header is exactly "learning_goal"
- **"CSV must contain level_X columns"**: Include at least one column named "level_0", "level_1", etc.
- **"No valid data found"**: Check that your CSV has data rows below the header

### Display Issues
- **Empty hierarchy**: Verify your CSV contains valid learning goals and numeric level values
- **Missing nodes**: Check for special characters or encoding issues in your CSV

### Save Issues
- **"No hierarchy data to save"**: Upload a CSV file first
- **"Save failed"**: Check your network connection and server status

## Future Enhancements

Planned features include:
- Import/export functionality for saved hierarchies
- Advanced filtering and search within hierarchies
- Integration with existing clustering analysis
- User management and sharing capabilities
- Custom level naming and configuration

## Support

For technical support or feature requests, please refer to the main application documentation or contact your system administrator. 